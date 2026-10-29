"""Exercise frozen-plan rejection and test-only evaluation with auditable masks."""

import contextlib
import copy
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from visual_lab.data.instruction_pairs import generate_dataset
from visual_lab.experiments.frozen_instruction_evaluation import validate_frozen_plan
from visual_lab.models import InstructionEditor, SemanticDilatedEditor
from visual_lab.runner import main


class FrozenInstructionChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.data = self.root / "data"
        with contextlib.redirect_stdout(io.StringIO()):
            generate_dataset(self.data, {"train": 1, "validation": 1, "test": 1})
        self.manifest = (self.data / "manifest.jsonl").read_bytes()
        self.semantic = self.root / "semantic"; self.semantic.mkdir()
        self.reference = self.root / "reference"; self.reference.mkdir()
        semantic, reference = SemanticDilatedEditor(), InstructionEditor()
        with torch.no_grad():
            for model in (semantic, reference):
                for parameter in model.parameters():parameter.zero_()
            semantic.layers[-1].bias[1] = 10  # Every pixel is classified as circle.
            reference.layers[-1].bias.fill_(10)  # Every requested mask is all true.
        torch.save({"model_state": semantic.state_dict()}, self.semantic / "model.pt")
        torch.save({"model_state": reference.state_dict()}, self.reference / "conditioned.pt")
        self.weights = (self.semantic / "model.pt").read_bytes()
        self.reference_weights = (self.reference / "conditioned.pt").read_bytes()
        digest = hashlib.sha256(self.manifest).hexdigest()
        self.plan = {"schema_version": 1, "split": "test", "selection_performed": False,
                     "frozen_before_first_test_performance": True, "expected_scenes": 1, "expected_requests": 2,
                     "semantic": {"model": "SemanticDilatedEditor", "architecture": "dilated",
                                  "dataset": "instruction_edits_v1", "selected_on": "validation",
                                  "test_model_performance_measured": False, "decoder": "three_class_argmax",
                                  "class_mapping": {"background": 0, "circle": 1, "square": 2},
                                  "checkpoint_sha256": hashlib.sha256(self.weights).hexdigest(), "manifest_sha256": digest},
                     "reference": {"model": "InstructionEditor", "dataset": "instruction_edits_v1",
                                   "selected_on": "validation", "test_model_performance_measured": False,
                                   "checkpoint_sha256": hashlib.sha256(self.reference_weights).hexdigest(),
                                   "manifest_sha256": digest, "mask_probability_threshold": .95}}

    def test_changed_weights_manifest_decoder_or_selection_are_rejected(self):
        validate_frozen_plan(self.plan, self.weights, self.reference_weights, self.manifest)
        for weights, reference, manifest in ((b"wrong", self.reference_weights, self.manifest),
                                              (self.weights, b"wrong", self.manifest),
                                              (self.weights, self.reference_weights, b"wrong")):
            with self.assertRaises(ValueError):validate_frozen_plan(self.plan, weights, reference, manifest)
        for key, value in (("selection_performed", True), ("split", "validation"), ("expected_requests", 3)):
            invalid = copy.deepcopy(self.plan); invalid[key] = value
            with self.assertRaises(ValueError):validate_frozen_plan(invalid, self.weights, self.reference_weights, self.manifest)
        invalid = copy.deepcopy(self.plan); invalid["semantic"]["decoder"] = "new_threshold"
        with self.assertRaises(ValueError):validate_frozen_plan(invalid, self.weights, self.reference_weights, self.manifest)

    def test_only_test_images_are_needed_and_saved_masks_reproduce_counts(self):
        for split in ("train", "validation"):shutil.rmtree(self.data / split)
        plan_path = self.root / "plan.json"; plan_path.write_text(json.dumps(self.plan))
        output = self.root / "output"
        config = {"schema_version": 1, "experiment": "frozen_fixture", "command": "evaluate_frozen_instruction_model",
                  "parameters": {"data_dir": str(self.data), "run_dir": str(self.semantic),
                                 "reference_run_dir": str(self.reference), "frozen_config": str(plan_path), "output_dir": str(output)}}
        path = self.root / "config.json"; path.write_text(json.dumps(config))
        with contextlib.redirect_stdout(io.StringIO()):main(["--config", str(path)])
        report = json.loads((output / "report.json").read_text())
        self.assertFalse(report["selection_performed"])
        self.assertFalse(report["training_performed"])
        self.assertEqual(report["requests"], 2)
        self.assertEqual(report["configuration"]["declared"], config)
        self.assertEqual(report["frozen_plan_sha256"], hashlib.sha256(plan_path.read_bytes()).hexdigest())
        self.assertEqual(report["saved_predictions_sha256"], hashlib.sha256((output / "predicted_masks.npz").read_bytes()).hexdigest())
        records = [json.loads(line) for line in self.manifest.decode().splitlines() if json.loads(line)["split"] == "test"]
        with np.load(output / "predicted_masks.npz") as predictions:
            self.assertTrue(predictions["semantic"][0].all())
            self.assertFalse(predictions["semantic"][1].any())
            for name in ("semantic", "reference"):
                wrong = 0
                for i, record in enumerate(records):
                    with Image.open(self.data / record["input"]) as image:pixels = np.array(image)
                    pixels[predictions[name][i]] = (255, 0, 0)
                    with Image.open(self.data / record["target"]) as image:wrong += int(np.any(pixels != np.array(image), axis=2).sum())
                counts = report["methods"]["frozen_" + name]["counts"]
                self.assertEqual(counts["pixels"] - counts["correct_pixels"], wrong)
                self.assertEqual(counts["exact_images"], 0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):main(["--config", str(path)])
