"""Hand-calculated checks for geometric grouping and boundary diagnostics."""

import unittest

import numpy as np

from analyze_spatial_errors import boundary_band, empty_pixel_gap


class GeometryChecks(unittest.TestCase):
    def test_adjacent_and_separated_pixel_gaps(self):
        first = np.zeros((8, 8), dtype=bool)
        second = first.copy()
        first[1, 1] = True
        second[1, 2] = True
        self.assertEqual(empty_pixel_gap(first, second), 0)
        second[1, 2] = False
        second[1, 5] = True
        self.assertEqual(empty_pixel_gap(first, second), 3)

    def test_boundary_band_distinguishes_edge_from_deep_interior(self):
        mask = np.zeros((15, 15), dtype=bool)
        mask[3:12, 3:12] = True
        band = boundary_band(mask, radius=2)
        self.assertTrue(band[3, 7])  # Object edge.
        self.assertTrue(band[2, 7])  # Just outside the edge.
        self.assertFalse(band[7, 7])  # Deep interior.
        self.assertFalse(band[0, 0])  # Distant background.


if __name__ == "__main__":
    unittest.main()
