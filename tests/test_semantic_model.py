"""Audit joint semantic supervision, loss gradients, compute, and held-out isolation."""

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
from visual_lab.experiments.semantic_training import (
    MODELS, collapse_scenes, convolution_macs, predict_scene_classes, request_logits, segmentation_loss,
)
from visual_lab.runner import load_configuration, main
from visual_lab.paths import ROOT
from visual_lab.semantic_decoding import semantic_object_masks


class SemanticChecks(unittest.TestCase):
    def test_fast_decoder_matches_argmax_including_every_three_way_tie_pattern(self):
        import itertools
        values = list(itertools.product((-1., 0., 1.), repeat=3))
        logits = torch.tensor(values).T.reshape(1, 3, 3, 9)
        classes = logits.argmax(1)
        expected = torch.stack((classes == 1, classes == 2), dim=1)
        torch.testing.assert_close(semantic_object_masks(logits), expected)
        random = torch.randn(4, 3, 16, 17)
        classes = random.argmax(1)
        torch.testing.assert_close(semantic_object_masks(random), torch.stack((classes == 1, classes == 2), dim=1))
        with self.assertRaises(ValueError):semantic_object_masks(torch.zeros(1, 2, 4, 4))

    def test_architectures_preserve_resolution_and_learn_all_classes(self):
        torch.set_num_threads(2)
        for name, constructor in MODELS.items():
            with self.subTest(architecture=name):
                torch.manual_seed(17)
                model = constructor()
                images = torch.rand(2, 3, 31, 33)
                logits = model(images)
                self.assertEqual(tuple(logits.shape), (2, 3, 31, 33))
                labels = torch.arange(31 * 33).reshape(1, 31, 33).repeat(2, 1, 1) % 3
                loss = segmentation_loss(logits, labels, torch.ones(3), .5)
                loss.backward()
                head = model.layers[-1] if name == "dilated" else model.head
                self.assertTrue((head.weight.grad.flatten(1).abs().sum(1) > 0).all())
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_perfect_prediction_has_lower_loss_for_both_objectives(self):
        labels = torch.tensor([[[0, 1, 2], [2, 1, 0]]])
        perfect = torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).float() * 30 - 15
        for dice in (0, .5):
            self.assertLess(segmentation_loss(perfect, labels, torch.ones(3), dice).item(), 1e-5)
            self.assertGreater(segmentation_loss(perfect.roll(1, 1), labels, torch.ones(3), dice).item(), 10)

    def test_instruction_routing_is_exclusive_and_uses_no_labels(self):
        classes = np.array([[[0, 1, 2], [1, 2, 0]]], dtype=np.uint8)
        records = [{"instruction_id": 0}, {"instruction_id": 1}]
        output = request_logits(classes, records, np.array([0, 0])) > 0
        np.testing.assert_array_equal(output[0], classes[0] == 1)
        np.testing.assert_array_equal(output[1], classes[0] == 2)
        self.assertFalse((output[0] & output[1]).any())

    def test_convolution_macs_count_groups_and_remove_hooks(self):
        for groups, expected in ((1, 3 * 3 * 3 * 6 * 4 * 5), (3, 3 * 3 * 1 * 6 * 4 * 5)):
            model = torch.nn.Conv2d(3, 6, 3, padding=1, groups=groups)
            self.assertEqual(convolution_macs(model, torch.zeros(1, 3, 4, 5)), expected)
            self.assertFalse(model._forward_hooks)

    def test_invalid_semantic_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = json.loads((ROOT / "configs/semantic_lite_ce_v1.json").read_text())
            path = Path(temporary) / "config.json"
            for key, value in (("dice_weight", -1), ("dice_weight", True), ("dice_weight", 2),
                               ("architecture", "unknown")):
                invalid = json.loads(json.dumps(config)); invalid["parameters"][key] = value
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    load_configuration(path)

    def test_semantic_training_reloads_without_test_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data_dir = root / "data"; reference = root / "reference"
            with contextlib.redirect_stdout(io.StringIO()):
                generate_dataset(data_dir, {"train": 4, "validation": 2, "test": 1})
            shutil.rmtree(data_dir / "test")
            manifest = (data_dir / "manifest.jsonl").read_bytes(); digest = hashlib.sha256(manifest).hexdigest()
            baseline_config = {"schema_version": 1, "experiment": "reference_fixture", "command": "train_instruction_model",
                               "parameters": {"epochs": 1, "batch_size": 4, "learning_rate": .001, "seed": 17,
                                              "device": "cpu", "mask_threshold": .5, "manifest_sha256": digest,
                                              "data_dir": str(data_dir), "output_dir": str(reference)}}
            path = root / "config.json"; path.write_text(json.dumps(baseline_config))
            with contextlib.redirect_stdout(io.StringIO()):main(["--config", str(path)])
            baseline = json.loads((reference / "validation.json").read_text())
            selected = root / "reference_config.json"
            selected.write_text(json.dumps({"model": "InstructionEditor", "selected_on": "validation",
                                           "manifest_sha256": digest, "checkpoint_sha256": baseline["checkpoint_sha256"]["conditioned"],
                                           "mask_probability_threshold": .95}))
            records = [json.loads(line) for line in manifest.decode().splitlines()]
            data = load_instruction_split(data_dir, records, "validation")
            collapsed = collapse_scenes(data)
            self.assertEqual(len(collapsed[0]), 2)
            self.assertEqual(set(torch.unique(collapsed[1]).tolist()), {0, 1, 2})
            modified = list(data); modified[2] = torch.ones_like(data[2])
            with self.assertRaisesRegex(ValueError, "disjoint"):collapse_scenes(modified)
            for architecture, dice in (("dilated", 0), ("lite_unet", .5)):
                output = root / architecture
                config = {"schema_version": 1, "experiment": "semantic_fixture", "command": "train_semantic_model",
                          "parameters": {"epochs": 2, "batch_size": 2, "learning_rate": .001, "seed": 17,
                                         "device": "cpu", "manifest_sha256": digest, "architecture": architecture,
                                         "dice_weight": dice, "data_dir": str(data_dir), "output_dir": str(output),
                                         "reference_run_dir": str(reference), "reference_config": str(selected)}}
                path.write_text(json.dumps(config))
                with contextlib.redirect_stdout(io.StringIO()):main(["--config", str(path)])
                report = json.loads((output / "validation.json").read_text())
                checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
                model = MODELS[architecture]().eval(); model.load_state_dict(checkpoint["model_state"], strict=True)
                classes = predict_scene_classes(model, collapsed[0], batch_size=1)
                self.assertEqual(report["configuration"]["declared"], config)
                self.assertEqual(report["checkpoint_sha256"], hashlib.sha256((output / "model.pt").read_bytes()).hexdigest())
                exact, wrong = [], 0
                for i, record in enumerate(data[3]):
                    pixels = data[0][i].numpy().copy()
                    pixels[classes[collapsed[2][i]] == record["instruction_id"] + 1] = (255, 0, 0)
                    with Image.open(data_dir / record["target"]) as image:errors = np.any(pixels != np.array(image), axis=2)
                    exact.append(not errors.any()); wrong += int(errors.sum())
                self.assertEqual(report["methods"]["semantic"]["counts"]["exact_images"], sum(exact))
                self.assertEqual(report["errors"]["wrong_pixels"], wrong)
                self.assertEqual(report["methods"]["semantic"]["scene_counts"]["exact_scenes"], sum(all(exact[i:i+2]) for i in (0, 2)))
                self.assertFalse(report["test_images_loaded"])
                self.assertFalse((data_dir / "test").exists())
                self.assertEqual(report["training_class_counts"], torch.bincount(collapse_scenes(load_instruction_split(data_dir, records, "train"))[1].flatten().long(), minlength=3).tolist())
            benchmark_output = root / "benchmark"
            config = {"schema_version": 1, "experiment": "decoder_fixture", "command": "benchmark_semantic_decoder",
                      "parameters": {"data_dir": str(data_dir), "run_dir": str(root / "dilated"),
                                     "reference_run_dir": str(reference), "reference_config": str(selected),
                                     "output_dir": str(benchmark_output)}}
            path.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):main(["--config", str(path)])
            benchmark_report = json.loads((benchmark_output / "report.json").read_text())
            self.assertEqual(benchmark_report["equivalent_request_masks"], 4)
            self.assertEqual(benchmark_report["equivalent_scene_predictions"], 2)
            self.assertFalse(benchmark_report["test_images_loaded"])
