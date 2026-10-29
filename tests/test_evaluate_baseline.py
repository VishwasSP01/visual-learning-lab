"""Hand-calculated checks that catch misleading evaluation results."""

import unittest

from PIL import Image

from visual_lab.baselines import recolor_blue
from visual_lab.metrics import score_prediction, summarize


class EvaluationChecks(unittest.TestCase):
    def setUp(self):
        # Three pixels: one edit, and two pixels that must stay unchanged.
        self.original = Image.new("RGB", (3, 1))
        self.original.putdata([(0, 0, 255), (0, 128, 0), (255, 255, 255)])
        self.target = Image.new("RGB", (3, 1))
        self.target.putdata([(255, 0, 0), (0, 128, 0), (255, 255, 255)])
        self.mask = Image.new("L", (3, 1))
        self.mask.putdata([255, 0, 0])

    def scores(self, prediction):
        return summarize(score_prediction(self.original, prediction, self.target, self.mask))

    def test_unchanged_image_has_no_successful_edit(self):
        scores = self.scores(self.original)
        self.assertEqual(scores["exact_image_success"], 0)
        self.assertEqual(scores["edit_pixel_accuracy"], 0)
        self.assertEqual(scores["background_preservation"], 1)
        self.assertAlmostEqual(scores["whole_image_pixel_accuracy"], 2 / 3)

    def test_recoloring_matches_the_complete_target(self):
        scores = self.scores(recolor_blue(self.original))
        self.assertEqual(scores["exact_image_success"], 1)
        self.assertEqual(scores["edit_pixel_accuracy"], 1)
        self.assertEqual(scores["background_preservation"], 1)
        self.assertEqual(self.original.getpixel((0, 0)), (0, 0, 255))

    def test_painting_everything_red_destroys_the_background(self):
        scores = self.scores(Image.new("RGB", (3, 1), "red"))
        self.assertEqual(scores["exact_image_success"], 0)
        self.assertEqual(scores["edit_pixel_accuracy"], 1)
        self.assertEqual(scores["background_preservation"], 0)
        self.assertAlmostEqual(scores["whole_image_pixel_accuracy"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
