"""Describe validation mistakes from the frozen CNN; do not retrain or tune it."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from train_spatial_model import ShapeEditor, predict_masks, render_edit


ROOT = Path(__file__).resolve().parent


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


def gap_group(gap):
    return "0-1" if gap <= 1 else "2-5" if gap <= 5 else "6+"


def summarize_group(cases):
    total = len(cases)
    return {
        "images": total,
        "failed_images": sum(case["error_pixels"] > 0 for case in cases),
        "exact_image_success": sum(case["error_pixels"] == 0 for case in cases) / total,
        "mean_error_pixels": sum(case["error_pixels"] for case in cases) / total,
        "error_pixels": sum(case["error_pixels"] for case in cases),
    }


def group_cases(cases, field):
    return {str(value): summarize_group([case for case in cases if case[field] == value])
            for value in sorted({case[field] for case in cases})}


def save_worst_preview(selected, data_dir, model, output):
    cell, row_height = 144, 168
    sheet = Image.new("RGB", (cell * 4, 54 + len(selected) * row_height), "#e8e8e8")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 5), "Worst four validation cases by wrong-pixel count (ties: example ID)", fill="black")
    for column, label in enumerate(("Input", "CNN edit", "Target", "Error locations")):
        draw.text((column * cell + 5, 25), label, fill="black")
    for row, case in enumerate(selected):
        with Image.open(data_dir / case["input"]) as image:
            original = image.copy()
        with Image.open(data_dir / case["target"]) as image:
            target = image.copy()
        with Image.open(data_dir / case["mask"]) as image:
            circle = np.array(image) == 255
        inputs = torch.from_numpy(np.array(original)).unsqueeze(0)
        predicted_mask = predict_masks(model, inputs, torch.device("cpu"))[0]
        prediction = render_edit(original, predicted_mask)
        overlay = np.array(Image.blend(target, Image.new("RGB", target.size, "white"), 0.65))
        overlay[predicted_mask & ~circle] = (255, 0, 255)
        overlay[~predicted_mask & circle] = (255, 190, 0)
        for column, image in enumerate((original, prediction, target, Image.fromarray(overlay))):
            sheet.paste(image.resize((cell, cell), Image.Resampling.NEAREST),
                        (column * cell, 44 + row * row_height))
        draw.text((5, 44 + row * row_height + cell + 3),
                  f"{case['id']} | wrong pixels: {case['error_pixels']} | shape gap: {case['gap']}", fill="black")
    draw.text((5, sheet.height - 15), "Magenta: unwanted edit. Gold: missed circle pixel. Other pixels are faded.", fill="black")
    sheet.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/shape_edits_v1")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/spatial_cnn")
    args = parser.parse_args()
    output = args.run_dir / "error_analysis"
    reference_bytes = (args.run_dir / "validation.json").read_bytes()
    reference = json.loads(reference_bytes)
    if reference["split"] != "validation" or reference["dataset"] != "shape_edits_v1":
        raise ValueError("This analysis is restricted to shape_edits_v1 validation data.")
    manifest = (args.data_dir / "manifest.jsonl").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != reference["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the recorded run.")
    records = [json.loads(line) for line in manifest.decode().splitlines()]
    records = [record for record in records if record["split"] == "validation"]
    checkpoint_path = args.run_dir / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ShapeEditor()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    torch.set_num_threads(4)
    cases = []
    for start in range(0, len(records), 32):
        batch = records[start:start + 32]
        loaded = []
        for record in batch:
            arrays = {}
            for kind in ("input", "target", "mask", "square_mask"):
                with Image.open(args.data_dir / record[kind]) as image:
                    arrays[kind] = np.array(image)
            loaded.append(arrays)
        inputs = torch.from_numpy(np.stack([images["input"] for images in loaded]))
        predictions = predict_masks(model, inputs, torch.device("cpu"))
        for record, images, prediction in zip(batch, loaded, predictions, strict=True):
            circle, square = images["mask"] == 255, images["square_mask"] == 255
            missed = circle & ~prediction
            unwanted = prediction & ~circle
            damaged_square = unwanted & square
            damaged_background = unwanted & ~square
            error = missed | unwanted
            near_boundary = boundary_band(circle) | boundary_band(square)
            ys, xs = np.nonzero(circle)
            radius = (int(xs.max()) - int(xs.min())) // 2
            edge_margin = min(int(xs.min()), int(ys.min()), 63 - int(xs.max()), 63 - int(ys.max()))
            gap = empty_pixel_gap(circle, square)
            # Verify mask-based diagnosis agrees with the actual rendered target.
            result = images["input"].copy()
            result[prediction] = (255, 0, 0)
            if not np.array_equal(np.any(result != images["target"], axis=2), error):
                raise ValueError(f"Mask errors disagree with rendered errors: {record['id']}")
            cases.append({
                "id": record["id"], "input": record["input"], "target": record["target"],
                "mask": record["mask"], "gap": gap, "gap_group": gap_group(gap),
                "circle_radius": radius, "circle_edge_margin": edge_margin,
                "edge_margin_group": "0-5" if edge_margin <= 5 else "6-10" if edge_margin <= 10 else "11+",
                "error_pixels": int(error.sum()), "missed_circle_pixels": int(missed.sum()),
                "damaged_square_pixels": int(damaged_square.sum()),
                "damaged_background_pixels": int(damaged_background.sum()),
                "errors_within_two_pixels_of_a_shape_boundary": int((error & near_boundary).sum()),
                "errors_farther_from_shape_boundaries": int((error & ~near_boundary).sum()),
            })

    counts = reference["methods"]["spatial_cnn"]["counts"]
    sums = {key: sum(case[key] for case in cases) for key in (
        "error_pixels", "missed_circle_pixels", "damaged_square_pixels", "damaged_background_pixels",
        "errors_within_two_pixels_of_a_shape_boundary", "errors_farther_from_shape_boundaries",
    )}
    expected = {
        "error_pixels": counts["pixels"] - counts["correct_pixels"],
        "missed_circle_pixels": counts["edited_pixels"] - counts["correct_edited_pixels"],
        "damaged_square_pixels": counts["square_pixels"] - counts["unchanged_square_pixels"],
        "damaged_background_pixels": counts["white_background_pixels"] - counts["unchanged_white_background_pixels"],
    }
    if any(sums[key] != value for key, value in expected.items()):
        raise ValueError("Reloaded model does not reproduce the reference pixel counts.")
    if len(cases) != counts["images"] or sum(case["error_pixels"] == 0 for case in cases) != counts["exact_images"]:
        raise ValueError("Reloaded model does not reproduce the reference image counts.")
    worst = sorted(cases, key=lambda case: (-case["error_pixels"], case["id"]))[:4]
    report = {
        "experiment": "shape_cnn_v1_error_analysis", "split": "validation",
        "images": len(cases), "exact_images": counts["exact_images"],
        "failed_images": len(cases) - counts["exact_images"],
        "error_totals": sums, "by_shape_gap": group_cases(cases, "gap_group"),
        "by_circle_radius": group_cases(cases, "circle_radius"),
        "by_circle_edge_margin": group_cases(cases, "edge_margin_group"),
        "worst_example_ids": [case["id"] for case in worst],
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "reference_report_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "manifest_sha256": reference["manifest_sha256"],
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_performed": False, "test_dataset_used": False,
        "gap_definition": "Minimum foreground Chebyshev distance minus one; adjacent pixels have gap 0.",
        "boundary_definition": "Dilation minus erosion with a 5x5 square neighborhood, for either shape.",
        "scope": "Exploratory associations on checkpoint-selection validation data; no causal or test-set claims.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "cases.jsonl").write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases), encoding="utf-8")
    save_worst_preview(worst, args.data_dir, model, output / "worst_cases.png")
    print(f"Audited {len(cases)} validation images; {report['failed_images']} contain errors.")
    print(json.dumps(sums, indent=2))
    print("Groups by shape gap:")
    print(json.dumps(report["by_shape_gap"], indent=2))
    print(f"Saved diagnosis and worst-case preview: {output}")


if __name__ == "__main__":
    main()
