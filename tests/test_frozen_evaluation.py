"""Ensure the test evaluator rejects a changed checkpoint or selection protocol."""

import hashlib
import unittest

from visual_lab.experiments.frozen_evaluation import validate_configuration


class FrozenConfigurationChecks(unittest.TestCase):
    def setUp(self):
        self.checkpoint = b"fixed checkpoint fixture"
        self.manifest = b"fixed manifest fixture"
        self.config = {
            "selected_on": "validation", "test_dataset_used": False,
            "dataset": "shape_edits_v1", "model": "ShapeEditor",
            "checkpoint_sha256": hashlib.sha256(self.checkpoint).hexdigest(),
            "manifest_sha256": hashlib.sha256(self.manifest).hexdigest(),
            "mask_probability_threshold": 0.8,
        }

    def test_frozen_configuration_is_accepted(self):
        self.assertEqual(validate_configuration(self.config, self.checkpoint, self.manifest), 0.8)

    def test_changed_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Checkpoint differs"):
            validate_configuration(self.config, b"other weights", self.manifest)

    def test_changed_manifest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            validate_configuration(self.config, self.checkpoint, b"other dataset")

    def test_test_selected_configuration_is_rejected(self):
        self.config["selected_on"] = "test"
        with self.assertRaisesRegex(ValueError, "without test data"):
            validate_configuration(self.config, self.checkpoint, self.manifest)


if __name__ == "__main__":
    unittest.main()
