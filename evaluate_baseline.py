"""Measure two simple reference methods before training a neural network."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent


def recolor_blue(image):
    """Use only the input pixels: replace exact blue with red everywhere."""
    result = image.copy()
    result.putdata([
        (255, 0, 0) if pixel == (0, 0, 255) else pixel
        for pixel in image.get_flattened_data()
    ])
    return result


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


def save_preview(examples, output):
    cell, row_height = 112, 128
    sheet = Image.new("RGB", (cell * 4, 24 + len(examples) * row_height), "#e8e8e8")
    drawing = ImageDraw.Draw(sheet)
    for column, title in enumerate(("Input", "Unchanged", "Color rule", "Target")):
        drawing.text((column * cell + 4, 6), title, fill="black")
    for row, images in enumerate(examples):
        for column, image in enumerate(images):
            enlarged = image.resize((cell, cell), Image.Resampling.NEAREST)
            sheet.paste(enlarged, (column * cell, 24 + row * row_height))
    sheet.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/circle_edits_v2")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/baselines")
    args = parser.parse_args()
    manifest_path = args.data_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        parser.error("Dataset missing. Run: uv run --locked python generate_pairs.py")
    manifest = manifest_path.read_bytes()
    records = [json.loads(line) for line in manifest.decode("utf-8").splitlines()]
    records = [record for record in records if record["split"] == args.split]
    if not records:
        parser.error(f"No examples found for split {args.split!r}.")

    methods = {"unchanged": lambda image: image.copy(), "color_rule": recolor_blue}
    totals = {method: {} for method in methods}
    data_hash = hashlib.sha256()
    examples = []
    for record in records:
        loaded = []
        for kind in ("input", "target", "mask"):
            path = args.data_dir / record[kind]
            data_hash.update(record[kind].encode("utf-8") + b"\0")
            data_hash.update(path.read_bytes())
            with Image.open(path) as image:
                loaded.append(image.copy())
        original, target, mask = loaded
        predictions = {}
        for name, method in methods.items():
            # The method cannot access the target, mask, or recorded geometry.
            prediction = method(original)
            predictions[name] = prediction
            counts = score_prediction(original, prediction, target, mask)
            for key, value in counts.items():
                totals[name][key] = totals[name].get(key, 0) + value
        if len(examples) < 3:
            examples.append((original, predictions["unchanged"], predictions["color_rule"], target))

    report = {
        "dataset": "circle_edits_v2", "split": args.split, "examples": len(records),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "evaluated_files_sha256": data_hash.hexdigest(),
        "metric": "exact RGB equality; pixel rates pooled across the selected split",
        "scope": "Fixed blue-to-red synthetic task; neither method learns or reads instructions.",
        "methods": {name: summarize(counts) for name, counts in totals.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{args.split}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_preview(examples, args.output_dir / f"{args.split}_preview.png")
    print(f"Evaluated {len(records):,} {args.split} images. No training performed.")
    for name, scores in report["methods"].items():
        print(
            f"{name}: exact images={scores['exact_image_success']:.2%}; "
            f"edit accuracy={scores['edit_pixel_accuracy']:.2%}; "
            f"background preserved={scores['background_preservation']:.2%}; "
            f"overall pixel accuracy={scores['whole_image_pixel_accuracy']:.2%}"
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
