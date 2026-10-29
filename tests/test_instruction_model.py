"""Check conditioning gradients, paired metrics, and training with no test files."""

import contextlib
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
from visual_lab.data.loaders import load_instruction_split
from visual_lab.experiments.instruction_training import counterfactual_logits, predict_logits
from visual_lab.experiments.instruction_thresholds import select_candidate, paired_changes
from visual_lab.instruction_metrics import instruction_counts, paired_scene_scores, summarize_requests
from visual_lab.models import InstructionEditor
from visual_lab.runner import main
from visual_lab.thresholds import apply_threshold


class InstructionMetricChecks(unittest.TestCase):
    def test_threshold_selection_prioritizes_complete_pairs_then_requests_errors_and_cutoff(self):
        def candidate(threshold, scenes, requests, errors):
            return {"threshold": threshold, "scores": {
                "scene_counts": {"exact_scenes": scenes},
                "counts": {"exact_images": requests, "pixels": 100, "correct_pixels": 100 - errors}}}
        best = candidate(0.7, 2, 5, 4)
        worse = [candidate(0.5, 1, 9, 0), candidate(0.5, 2, 4, 0),
                 candidate(0.5, 2, 5, 5), candidate(0.9, 2, 5, 4)]
        self.assertIs(select_candidate([*worse, best]), best)
        self.assertIs(select_candidate([best, *reversed(worse)]), best)
        with self.assertRaises(ValueError):
            select_candidate([])

    def test_threshold_changes_count_pair_regressions_separately(self):
        before = [{"id": str(i), "scene_id": str(i // 2), "instruction_id": i % 2,
                   "exact": exact, "wrong_pixels": 0 if exact else 3}
                  for i, exact in enumerate((True, True, False, True))]
        after = [{**case, "exact": exact, "wrong_pixels": 0 if exact else 2}
                 for case, exact in zip(before, (True, False, True, True), strict=True)]
        changes = paired_changes(before, after)
        self.assertEqual(changes["scene_pairs_became_exact"], 1)
        self.assertEqual(changes["scene_pairs_lost_exactness"], 1)
        self.assertEqual(changes["requests_became_exact"], 1)
        self.assertEqual(changes["requests_lost_exactness"], 1)
        with self.assertRaises(ValueError):
            paired_changes(before, list(reversed(after)))

    def test_square_edit_measures_circle_preservation(self):
        original = Image.new("RGB", (3, 1))
        original.putdata([(0, 0, 255), (0, 0, 255), (255, 255, 255)])
        target = original.copy()
        target.putpixel((1, 0), (255, 0, 0))
        edit, preserve = Image.new("L", (3, 1)), Image.new("L", (3, 1))
        edit.putdata([0, 255, 0])
        preserve.putdata([255, 0, 0])
        scores = summarize_requests(instruction_counts(original, target, target, edit, preserve))
        self.assertEqual(scores["exact_image_success"], 1)
        self.assertEqual(scores["opposite_object_preservation"], 1)
        broken = summarize_requests(instruction_counts(original, Image.new("RGB", (3, 1), "red"), target, edit, preserve))
        self.assertEqual(broken["edit_pixel_accuracy"], 1)
        self.assertEqual(broken["opposite_object_preservation"], 0)
        self.assertEqual(broken["white_background_preservation"], 0)
        self.assertEqual(broken["exact_image_success"], 0)

    def test_both_requests_must_be_exact_for_scene_success(self):
        cases = [{"scene_id": scene, "instruction_id": task, "exact": exact}
                 for scene, task, exact in (("a", 0, True), ("a", 1, True),
                                            ("b", 0, True), ("b", 1, False))]
        scores = paired_scene_scores(cases)
        self.assertEqual(scores["paired_scene_success"], 0.5)
        self.assertEqual(scores["scene_counts"], {"scenes": 2, "exact_scenes": 1})
        with self.assertRaises(ValueError):
            paired_scene_scores(cases[:-1])

    def test_always_circle_output_cannot_complete_a_scene_pair(self):
        scores = paired_scene_scores([
            {"scene_id": "a", "instruction_id": 0, "exact": True},
            {"scene_id": "a", "instruction_id": 1, "exact": False},
        ])
        self.assertEqual(scores["paired_scene_success"], 0)


class ConditioningChecks(unittest.TestCase):
    def test_embedding_receives_gradients_for_both_tasks(self):
        torch.manual_seed(5)
        model = InstructionEditor()
        images = torch.rand(2, 3, 16, 16)
        output = model(images, torch.tensor([0, 1]))
        self.assertEqual(tuple(output.shape), (2, 1, 16, 16))
        output.square().mean().backward()
        gradient = model.instruction_embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient.abs().sum(dim=1) > 0).all())

    def test_counterfactuals_match_direct_inference_and_ignore_label_masks(self):
        torch.manual_seed(5)
        model = InstructionEditor().eval()
        images = torch.randint(0, 256, (1, 16, 16, 3), dtype=torch.uint8).repeat(2, 1, 1, 1)
        records = [{"scene_id": "same", "instruction_id": task} for task in (0, 1)]
        data = (images, torch.tensor([0, 1]), torch.zeros(2, 16, 16, dtype=torch.uint8), records, "fixture")
        logits = predict_logits(model, data, batch_size=2)
        reversed_data = (images, torch.tensor([1, 0]), data[2], records, "fixture")
        np.testing.assert_array_equal(counterfactual_logits(logits, records, swap=True),
                                      predict_logits(model, reversed_data, batch_size=2))
        clamped = predict_logits(model, data, batch_size=2, conditioned=False)
        np.testing.assert_array_equal(clamped[0], clamped[1])
        np.testing.assert_array_equal(counterfactual_logits(logits, records), clamped)
        changed_labels = (images, data[1], torch.ones_like(data[2]), records, "fixture")
        np.testing.assert_array_equal(logits, predict_logits(model, changed_labels, batch_size=2))


class InstructionTrainingChecks(unittest.TestCase):
    def test_config_training_reloads_and_scores_without_any_test_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir, output = root / "data", root / "output"
            with contextlib.redirect_stdout(io.StringIO()):
                generate_dataset(data_dir, {"train": 4, "validation": 2, "test": 1})
            shutil.rmtree(data_dir / "test")
            manifest = (data_dir / "manifest.jsonl").read_bytes()
            records = [json.loads(line) for line in manifest.decode().splitlines()]
            with self.assertRaisesRegex(ValueError, "test remains reserved"):
                load_instruction_split(data_dir, records, "test")
            config = {"schema_version": 1, "experiment": "instruction_fixture", "command": "train_instruction_model",
                      "parameters": {"epochs": 2, "batch_size": 4, "learning_rate": 0.001,
                                     "device": "cpu", "seed": 17, "mask_threshold": 0.5,
                                     "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                                     "data_dir": str(data_dir), "output_dir": str(output)}}
            path = root / "config.json"
            path.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--config", str(path)])
            report = json.loads((output / "validation.json").read_text())
            self.assertEqual(report["train_scenes"], 4)
            self.assertEqual(report["train_requests"], 8)
            self.assertEqual(report["validation_scenes"], 2)
            self.assertEqual(report["validation_requests"], 4)
            self.assertEqual(report["seed"], 17)
            self.assertFalse(report["test_images_loaded"])
            self.assertFalse((data_dir / "test").exists())
            self.assertEqual(report["configuration"]["declared"], config)
            self.assertEqual(report["experiment"], config["experiment"])
            self.assertEqual(report["training"]["conditioned"]["initial_state_sha256"],
                             report["training"]["instruction_blind"]["initial_state_sha256"])
            for key in ("instruction_blind", "conditioned_clamped_to_circle"):
                self.assertEqual(report["methods"][key]["paired_scene_success"], 0)
                self.assertLessEqual(report["methods"][key]["exact_image_success"], 0.5)
            checkpoint = torch.load(output / "conditioned.pt", map_location="cpu", weights_only=True)
            model = InstructionEditor().eval()
            model.load_state_dict(checkpoint["model_state"], strict=True)
            data = load_instruction_split(data_dir, records, "validation")
            predictions = predict_logits(model, data, batch_size=4) >= 0
            wrong, exact = 0, 0
            for index, record in enumerate(data[3]):
                pixels = data[0][index].numpy().copy()
                pixels[predictions[index]] = (255, 0, 0)
                with Image.open(data_dir / record["target"]) as image:
                    errors = np.any(pixels != np.array(image), axis=2)
                wrong += int(errors.sum())
                exact += int(not errors.any())
            counts = report["methods"]["conditioned"]["counts"]
            self.assertEqual(counts["exact_images"], exact)
            self.assertEqual(counts["pixels"] - counts["correct_pixels"], wrong)
            self.assertEqual(report["checkpoint_sha256"]["conditioned"], hashlib.sha256((output / "conditioned.pt").read_bytes()).hexdigest())

            # Sweep the saved fixture model with the entire test directory absent.
            sweep_output = root / "sweep"
            sweep_config = {"schema_version": 1, "experiment": "threshold_fixture",
                            "command": "compare_instruction_thresholds", "parameters": {
                                "data_dir": str(data_dir), "run_dir": str(output),
                                "output_dir": str(sweep_output)}}
            path.write_text(json.dumps(sweep_config))
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--config", str(path)])
            sweep = json.loads((sweep_output / "report.json").read_text())
            self.assertEqual(sweep["candidates"][0]["scores"], report["methods"]["conditioned"])
            self.assertFalse(sweep["training_performed"])
            self.assertFalse(sweep["test_images_loaded"])
            self.assertEqual(sweep["configuration"]["declared"], sweep_config)
            logits = predict_logits(model, data, batch_size=4)
            for candidate in sweep["candidates"]:
                exact, errors = [], {"missed_requested_pixels": 0, "damaged_opposite_pixels": 0,
                                     "damaged_background_pixels": 0, "wrong_pixels": 0}
                for index, record in enumerate(data[3]):
                    edited = apply_threshold(logits[index], candidate["threshold"])
                    pixels = data[0][index].numpy().copy()
                    pixels[edited] = (255, 0, 0)
                    with Image.open(data_dir / record["target"]) as image:
                        wrong = np.any(pixels != np.array(image), axis=2)
                    with Image.open(data_dir / record["preserve_mask"]) as image:
                        opposite = np.array(image) == 255
                    requested = data[2][index].numpy().astype(bool)
                    errors["missed_requested_pixels"] += int((wrong & requested).sum())
                    errors["damaged_opposite_pixels"] += int((wrong & opposite).sum())
                    errors["damaged_background_pixels"] += int((wrong & ~(requested | opposite)).sum())
                    errors["wrong_pixels"] += int(wrong.sum())
                    exact.append(not wrong.any())
                self.assertEqual(candidate["errors"], errors)
                self.assertEqual(candidate["scores"]["counts"]["exact_images"], sum(exact))
                self.assertEqual(candidate["scores"]["scene_counts"]["exact_scenes"],
                                 sum(all(exact[i:i + 2]) for i in range(0, len(exact), 2)))
            self.assertFalse((data_dir / "test").exists())
            self.assertEqual(sweep["checkpoint_sha256"], report["checkpoint_sha256"]["conditioned"])
            from visual_lab.experiments.instruction_thresholds import main as sweep_main
            flags = ["--data-dir", str(data_dir), "--run-dir", str(output),
                     "--output-dir", str(sweep_output)]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sweep_main(flags)
            original_checkpoint = (output / "conditioned.pt").read_bytes()
            (output / "conditioned.pt").write_bytes(original_checkpoint + b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint differs"):
                sweep_main(flags[:-1] + [str(root / "tampered_sweep")])
