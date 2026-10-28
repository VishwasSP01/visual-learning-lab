"""Check confidence decisions and the predeclared candidate-selection rule."""

import unittest

import numpy as np

from compare_thresholds import apply_threshold, select_candidate


class ThresholdChecks(unittest.TestCase):
    def test_logit_cutoff_preserves_expected_decisions(self):
        logits = np.array([-3.0, 0.0, 2.0, 4.0])
        np.testing.assert_array_equal(apply_threshold(logits, 0.5), [False, True, True, True])
        np.testing.assert_array_equal(apply_threshold(logits, 0.9), [False, False, False, True])

    def test_invalid_probability_threshold_is_rejected(self):
        for threshold in (0, 1, -0.1):
            with self.assertRaises(ValueError):
                apply_threshold(np.array([0.0]), threshold)

    def test_selection_prioritizes_exact_images_then_errors_then_lower_threshold(self):
        def candidate(threshold, exact, wrong):
            return {"threshold": threshold, "scores": {"counts": {
                "exact_images": exact, "pixels": 100, "correct_pixels": 100 - wrong,
            }}}
        candidates = [candidate(0.5, 5, 2), candidate(0.7, 6, 6),
                      candidate(0.9, 6, 4), candidate(0.95, 6, 4)]
        self.assertEqual(select_candidate(candidates)["threshold"], 0.9)


if __name__ == "__main__":
    unittest.main()
