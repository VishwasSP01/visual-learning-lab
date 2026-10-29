"""Geometric measurements for validation error analysis."""

import numpy as np


def boundary_band(mask, radius=2):
    """Pixels within a square-neighborhood radius of either side of a boundary."""
    height, width = mask.shape
    padded = np.pad(mask, radius, constant_values=False)
    neighborhoods = [padded[y:y + height, x:x + width]
                     for y in range(2 * radius + 1) for x in range(2 * radius + 1)]
    return np.logical_or.reduce(neighborhoods) & ~np.logical_and.reduce(neighborhoods)


def empty_pixel_gap(first, second):
    """Minimum Chebyshev foreground distance minus one; adjacent pixels have gap 0."""
    first_pixels = np.argwhere(first)
    second_pixels = np.argwhere(second)
    if len(first_pixels) == 0 or len(second_pixels) == 0 or (first & second).any():
        raise ValueError("Expected separate, non-empty shapes.")
    distances = np.abs(first_pixels[:, None, :] - second_pixels[None, :, :]).max(axis=2)
    return int(distances.min()) - 1
