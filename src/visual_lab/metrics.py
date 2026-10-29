"""Exact-image, edit-region, and preservation metrics."""

import numpy as np


def score_prediction(original, prediction, target, mask):
    """Count exact pixel matches; masks are used by the evaluator only."""
    if any(image.mode != "RGB" for image in (original, prediction, target)):
        raise ValueError("Input, prediction, and target must be RGB images.")
    if mask.mode != "L" or any(
        image.size != original.size for image in (prediction, target, mask)
    ):
        raise ValueError("Image sizes must agree and the mask must be grayscale.")
    if set(mask.get_flattened_data()) != {0, 255}:
        raise ValueError("A binary mask must include an edit region and background.")

    counts = {
        "images": 1, "exact_images": 0,
        "edited_pixels": 0, "correct_edited_pixels": 0,
        "background_pixels": 0, "unchanged_background_pixels": 0,
        "pixels": original.width * original.height, "correct_pixels": 0,
    }
    for before, actual, expected, selected in zip(
        original.get_flattened_data(), prediction.get_flattened_data(),
        target.get_flattened_data(), mask.get_flattened_data(), strict=True,
    ):
        matches_target = actual == expected
        counts["correct_pixels"] += int(matches_target)
        if selected == 255:
            counts["edited_pixels"] += 1
            counts["correct_edited_pixels"] += int(matches_target)
        else:
            counts["background_pixels"] += 1
            counts["unchanged_background_pixels"] += int(actual == before)
    counts["exact_images"] = int(counts["correct_pixels"] == counts["pixels"])
    return counts


def summarize(counts):
    """Pixel rates are pooled across images; image success requires every pixel."""
    return {
        "counts": counts,
        "exact_image_success": counts["exact_images"] / counts["images"],
        "edit_pixel_accuracy": counts["correct_edited_pixels"] / counts["edited_pixels"],
        "background_preservation": (
            counts["unchanged_background_pixels"] / counts["background_pixels"]
        ),
        "whole_image_pixel_accuracy": counts["correct_pixels"] / counts["pixels"],
    }


def spatial_counts(original, prediction, target, mask, square_mask):
    """Measure the square separately so the white background cannot hide damage."""
    counts = score_prediction(original, prediction, target, mask)
    before, after = np.array(original), np.array(prediction)
    circle = np.array(mask) == 255
    square = np.array(square_mask) == 255
    if square_mask.mode != "L" or square.shape != circle.shape or not square.any() or (circle & square).any():
        raise ValueError("Invalid square-preservation mask.")
    same = np.all(before == after, axis=2)
    changed = np.any(before != after, axis=2)
    white_background = ~(circle | square)
    counts.update({
        "square_pixels": int(square.sum()),
        "unchanged_square_pixels": int((same & square).sum()),
        "exact_squares": int(same[square].all()),
        "white_background_pixels": int(white_background.sum()),
        "unchanged_white_background_pixels": int((same & white_background).sum()),
        "mask_intersection": int((changed & circle).sum()),
        "mask_union": int((changed | circle).sum()),
    })
    return counts


def summarize_spatial(counts):
    return {
        **summarize(counts),
        "square_preservation": counts["unchanged_square_pixels"] / counts["square_pixels"],
        "exact_square_preservation": counts["exact_squares"] / counts["images"],
        "white_background_preservation": counts["unchanged_white_background_pixels"] / counts["white_background_pixels"],
        "edit_mask_iou": counts["mask_intersection"] / counts["mask_union"],
    }
