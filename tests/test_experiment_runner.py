"""Exercise config errors, output protection, and seeded training end to end."""

import contextlib
import hashlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

from visual_lab.data.spatial_pairs import make_pair
from visual_lab.paths import ROOT
from visual_lab.runner import load_configuration, main


class ConfigurationChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "nested/experiment.json"
        self.path.parent.mkdir()
        self.config = {
            "schema_version": 1, "experiment": "baseline_fixture",
            "command": "evaluate_baseline",
            "parameters": {"data_dir": "data/fixture", "split": "validation",
                           "output_dir": str(self.root / "output")},
        }

    def write(self):
        self.path.write_text(json.dumps(self.config))

    def test_all_checked_in_configs_validate(self):
        paths = list((ROOT / "configs").glob("*.json"))
        self.assertEqual(len(paths), 15)
        for path in paths:
            with self.subTest(config=path.name):
                _, parameters, context = load_configuration(path)
                self.assertTrue(Path(parameters["data_dir"] if "data_dir" in parameters
                                     else parameters["run_dir"]).is_absolute())
                self.assertEqual(context["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_paths_resolve_from_repository_not_config_directory(self):
        self.write()
        _, parameters, _ = load_configuration(self.path, root=self.root)
        self.assertEqual(parameters["data_dir"], str((self.root / "data/fixture").resolve()))

    def test_typo_or_missing_option_is_rejected(self):
        self.config["parameters"]["splti"] = self.config["parameters"].pop("split")
        self.write()
        with self.assertRaisesRegex(ValueError, "Parameters"):
            load_configuration(self.path)

    def test_duplicate_keys_and_nonfinite_numbers_are_rejected(self):
        for raw in ('{"schema_version":1,"schema_version":1}', '{"value":NaN}'):
            self.path.write_text(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                load_configuration(self.path)

    def test_invalid_training_settings_are_rejected(self):
        template = json.loads((ROOT / "configs/shape_cnn_v1.json").read_text())
        for key, value in (("seed", True), ("seed", -1), ("seed", 2**32),
                           ("batch_size", 0), ("epochs", 1.5),
                           ("learning_rate", -0.1), ("learning_rate", 1e309),
                           ("device", "cuda")):
            with self.subTest(key=key, value=value):
                self.config = json.loads(json.dumps(template))
                self.config["parameters"][key] = value
                self.write()
                with self.assertRaises(ValueError):
                    load_configuration(self.path)

    def test_dry_run_does_not_create_output_or_require_data(self):
        self.write()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(["--config", str(self.path), "--dry-run"])
        self.assertEqual(json.loads(output.getvalue())["declared"], self.config)
        self.assertFalse(Path(self.config["parameters"]["output_dir"]).exists())

    def test_existing_output_is_preserved_before_data_is_loaded(self):
        self.write()
        output = Path(self.config["parameters"]["output_dir"])
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("previous result")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--config", str(self.path)])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(sentinel.read_text(), "previous result")


class TrainingConfigurationChecks(unittest.TestCase):
    def test_shuffle_seed_is_independent_of_global_rng(self):
        from visual_lab.experiments.pixel_training import make_loader as pixel_loader
        from visual_lab.experiments.spatial_training import make_loader as spatial_loader

        labels = torch.arange(32)
        for loader in (pixel_loader, spatial_loader):
            with self.subTest(loader=loader.__module__):
                orders = []
                for seed in (23, 23, 24):
                    torch.rand(17)  # Unrelated model RNG use must not alter sample order.
                    orders.append(torch.cat([batch[1] for batch in loader(
                        labels, labels, 5, shuffle=True, seed=seed,
                    )]))
                self.assertTrue(torch.equal(orders[0], orders[1]))
                self.assertFalse(torch.equal(orders[0], orders[2]))

    def test_config_seed_controls_both_models_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data"
            dataset.mkdir()
            records = []
            # No test split exists: training must only need train and validation.
            for split, count in (("train", 4), ("validation", 3)):
                rng = random.Random(111 if split == "train" else 112)
                for index in range(count):
                    images, _ = make_pair(rng, split)
                    record = {"split": split, "id": f"{split}-{index}"}
                    for kind, image in zip(("input", "target", "mask", "square_mask"), images, strict=True):
                        record[kind] = f"{split}-{index}-{kind}.png"
                        image.save(dataset / record[kind])
                    records.append(record)
            (dataset / "manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
            for command in ("train_pixel_model", "train_spatial_model"):
                with self.subTest(command=command):
                    states = []
                    for index, seed in enumerate((23, 23, 24)):
                        output = root / f"{command}-{index}"
                        config = {"schema_version": 1, "experiment": "training_fixture",
                                  "command": command, "parameters": {
                                      "epochs": 1, "batch_size": 2, "learning_rate": 0.001,
                                      "device": "cpu", "seed": seed, "data_dir": str(dataset),
                                      "output_dir": str(output),
                                  }}
                        path = root / "training.json"
                        path.write_text(json.dumps(config))
                        with contextlib.redirect_stdout(io.StringIO()):
                            main(["--config", str(path)])
                        report = json.loads((output / "validation.json").read_text())
                        checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
                        self.assertEqual(report["seed"], seed)
                        self.assertEqual(checkpoint["seed"], seed)
                        self.assertEqual(report["effective_parameters"]["seed"], seed)
                        self.assertEqual(report["configuration"]["declared"], config)
                        self.assertEqual(report["configuration"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                        self.assertIn("git_commit", report)
                        self.assertIn("git_dirty", report)
                        self.assertEqual(report["train_examples"], 4)
                        self.assertEqual(report["validation_examples"], 3)
                        self.assertIn("uv.lock", report["environment_files_sha256"])
                        self.assertIn("src/visual_lab/models.py", report["source_files_sha256"])
                        states.append(checkpoint["model_state"])
                    self.assertTrue(all(torch.equal(states[0][key], states[1][key]) for key in states[0]))
                    self.assertTrue(any(not torch.equal(states[0][key], states[2][key]) for key in states[0]))


if __name__ == "__main__":
    unittest.main()
