"""Generic preservation metrics and exact success under both instructions."""

from collections import defaultdict

from visual_lab.metrics import spatial_counts, summarize


def instruction_counts(original, prediction, target, edit_mask, preserve_mask):
    # The existing spatial metric measures any edit region and disjoint distractor.
    # Replace its historical square-specific names for either preservation task.
    counts = spatial_counts(original, prediction, target, edit_mask, preserve_mask)
    for old, new in (("square_pixels", "preserved_object_pixels"),
                     ("unchanged_square_pixels", "unchanged_object_pixels"),
                     ("exact_squares", "exact_preserved_objects")):
        counts[new] = counts.pop(old)
    return counts


def summarize_requests(counts):
    return {
        **summarize(counts),
        "opposite_object_preservation": counts["unchanged_object_pixels"] / counts["preserved_object_pixels"],
        "exact_opposite_object_preservation": counts["exact_preserved_objects"] / counts["images"],
        "white_background_preservation": counts["unchanged_white_background_pixels"] / counts["white_background_pixels"],
        "edit_mask_iou": counts["mask_intersection"] / counts["mask_union"],
    }


def paired_scene_scores(cases):
    groups = defaultdict(list)
    for case in cases:
        groups[case["scene_id"]].append(case)
    if not groups:
        raise ValueError("No scene pairs to score.")
    exact = 0
    for pair in groups.values():
        if len(pair) != 2 or {case["instruction_id"] for case in pair} != {0, 1}:
            raise ValueError("Scene scores require one result for each instruction.")
        exact += int(all(case["exact"] for case in pair))
    return {"scene_counts": {"scenes": len(groups), "exact_scenes": exact},
            "paired_scene_success": exact / len(groups)}
