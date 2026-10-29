"""Threshold decisions and the predeclared validation selection rule."""

import math


THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95)
SELECTION_RULE = "Most exact validation images; then fewest wrong pixels; then lowest threshold."


def apply_threshold(logits, threshold):
    """Compare logits directly, equivalent to sigmoid(logits) >= threshold."""
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be strictly between 0 and 1.")
    return logits >= math.log(threshold / (1 - threshold))


def select_candidate(candidates):
    if not candidates:
        raise ValueError("No candidates to select.")
    return max(candidates, key=lambda candidate: (
        candidate["scores"]["counts"]["exact_images"],
        -(candidate["scores"]["counts"]["pixels"] - candidate["scores"]["counts"]["correct_pixels"]),
        -candidate["threshold"],
    ))
