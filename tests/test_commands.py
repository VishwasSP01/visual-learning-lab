"""Check that installed-package commands work without local data or working-directory imports."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CommandChecks(unittest.TestCase):
    def test_experiment_help_works_outside_the_repository(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        commands = (
            "generate_pairs", "generate_spatial_pairs", "evaluate_baseline",
            "train_pixel_model", "train_spatial_model", "inspect_pixel_model",
            "analyze_spatial_errors", "compare_thresholds", "evaluate_frozen_model",
        )
        with tempfile.TemporaryDirectory() as working_directory:
            for name in commands:
                with self.subTest(command=name):
                    result = subprocess.run(
                        [sys.executable, str(scripts / f"{name}.py"), "--help"],
                        cwd=working_directory, capture_output=True, text=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
