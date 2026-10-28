"""Check that incorrect square edits cannot hide behind white background pixels."""

import unittest

from PIL import Image

from evaluate_baseline import recolor_blue
from train_spatial_model import spatial_counts, summarize_spatial


class SpatialMetricChecks(unittest.TestCase):
    def setUp(self):
        # A circle pixel, a square pixel, and one white background pixel.
        self.original = Image.new("RGB", (3, 1))
        self.original.putdata([(0, 0, 255), (0, 0, 255), (255, 255, 255)])
        self.target = Image.new("RGB", (3, 1))
        self.target.putdata([(255, 0, 0), (0, 0, 255), (255, 255, 255)])
        self.circle = Image.new("L", (3, 1))
        self.circle.putdata([255, 0, 0])
        self.square = Image.new("L", (3, 1))
        self.square.putdata([0, 255, 0])

    def scores(self, prediction):
        return summarize_spatial(spatial_counts(
            self.original, prediction, self.target, self.circle, self.square,
        ))

    def test_rule_fails_square_despite_perfect_circle_and_white_background(self):
        scores = self.scores(recolor_blue(self.original))
        self.assertEqual(scores["edit_pixel_accuracy"], 1)
        self.assertEqual(scores["square_preservation"], 0)
        self.assertEqual(scores["white_background_preservation"], 1)
        self.assertEqual(scores["exact_image_success"], 0)
        self.assertEqual(scores["edit_mask_iou"], 0.5)
        self.assertEqual(scores["background_preservation"], 0.5)

    def test_correct_edit_preserves_square(self):
        scores = self.scores(self.target)
        self.assertEqual(scores["square_preservation"], 1)
        self.assertEqual(scores["exact_square_preservation"], 1)
        self.assertEqual(scores["edit_mask_iou"], 1)
        self.assertEqual(scores["exact_image_success"], 1)


if __name__ == "__main__":
    unittest.main()
