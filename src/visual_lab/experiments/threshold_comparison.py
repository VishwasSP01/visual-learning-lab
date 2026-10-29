"""Compare a fixed confidence-threshold grid on validation data without retraining."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from visual_lab.metrics import spatial_counts, summarize_spatial
from visual_lab.models import ShapeEditor
from visual_lab.paths import ROOT
from visual_lab.provenance import source_hashes
from visual_lab.rendering import render_edit
from visual_lab.thresholds import THRESHOLDS, SELECTION_RULE, apply_threshold, select_candidate


def save_preview(indices, records, images, logits, threshold, output):
    cell, row_height = 128, 152
    sheet = Image.new("RGB", (4 * cell, 24 + len(indices) * row_height), "#e8e8e8")
    drawing = ImageDraw.Draw(sheet)
    for column, title in enumerate(("Input", "Threshold 0.50", f"Selected {threshold:.2f}", "Target")):
        drawing.text((column * cell + 4, 6), title, fill="black")
    for row, index in enumerate(indices):
        original = images[index]["input"]
        panels = (original, render_edit(original, apply_threshold(logits[index], 0.5)),
                  render_edit(original, apply_threshold(logits[index], threshold)), images[index]["target"])
        for column, panel in enumerate(panels):
            sheet.paste(panel.resize((cell, cell), Image.Resampling.NEAREST),
                        (column * cell, 24 + row * row_height))
        drawing.text((5, 24 + row * row_height + cell + 4), records[index]["id"], fill="black")
    sheet.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/shape_edits_v1")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/spatial_cnn")
    args = parser.parse_args()
    output = args.run_dir / "threshold_sweep"
    reference_bytes = (args.run_dir / "validation.json").read_bytes()
    reference = json.loads(reference_bytes)
    if reference["split"] != "validation" or reference["dataset"] != "shape_edits_v1":
        raise ValueError("This experiment only selects thresholds on shape_edits_v1 validation data.")
    manifest = (args.data_dir / "manifest.jsonl").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != reference["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the reference run.")
    records = [json.loads(line) for line in manifest.decode().splitlines()]
    records = [record for record in records if record["split"] == "validation"]
    checkpoint_path = args.run_dir / "model.pt"
    checkpoint_before = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ShapeEditor()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    torch.set_num_threads(4)

    images = []
    evaluated_files = hashlib.sha256()
    for record in records:
        example = {}
        for kind in ("input", "target", "mask", "square_mask"):
            path = args.data_dir / record[kind]
            evaluated_files.update(record[kind].encode() + b"\0" + path.read_bytes())
            with Image.open(path) as image:
                example[kind] = image.copy()
        images.append(example)

    # Run inference once; all candidates use exactly the same frozen logits.
    batches = []
    with torch.no_grad():
        for start in range(0, len(images), 32):
            batch = np.stack([np.array(image["input"]) for image in images[start:start + 32]])
            inputs = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255
            batches.append(model(inputs)[:, 0].numpy())
    logits = np.concatenate(batches)
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite model output.")

    candidates, per_case = [], {}
    previous_mask = None
    for threshold in THRESHOLDS:
        masks = apply_threshold(logits, threshold)
        if previous_mask is not None and (masks & ~previous_mask).any():
            raise ValueError("A higher threshold unexpectedly introduced new edit pixels.")
        previous_mask = masks
        totals, cases = {}, []
        for record, example, predicted_mask in zip(records, images, masks, strict=True):
            prediction = render_edit(example["input"], predicted_mask)
            counts = spatial_counts(example["input"], prediction, example["target"],
                                    example["mask"], example["square_mask"])
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
            cases.append({
                "id": record["id"], "threshold": threshold,
                "exact": bool(counts["exact_images"]),
                "wrong_pixels": counts["pixels"] - counts["correct_pixels"],
                "missed_circle_pixels": counts["edited_pixels"] - counts["correct_edited_pixels"],
                "damaged_square_pixels": counts["square_pixels"] - counts["unchanged_square_pixels"],
                "damaged_background_pixels": counts["white_background_pixels"] - counts["unchanged_white_background_pixels"],
            })
        candidates.append({"threshold": threshold, "scores": summarize_spatial(totals)})
        per_case[str(threshold)] = cases

    if candidates[0]["scores"]["counts"] != reference["methods"]["spatial_cnn"]["counts"]:
        raise ValueError("Threshold 0.50 does not reproduce the original run's validation counts.")
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_before:
        raise ValueError("Checkpoint changed during the experiment.")
    selected = select_candidate(candidates)
    selected_threshold = selected["threshold"]
    baseline_cases = per_case["0.5"]
    selected_cases = per_case[str(selected_threshold)]
    paired = {
        "images_with_fewer_wrong_pixels": sum(b["wrong_pixels"] < a["wrong_pixels"] for a, b in zip(baseline_cases, selected_cases, strict=True)),
        "images_with_more_wrong_pixels": sum(b["wrong_pixels"] > a["wrong_pixels"] for a, b in zip(baseline_cases, selected_cases, strict=True)),
        "images_with_equal_wrong_pixels": sum(b["wrong_pixels"] == a["wrong_pixels"] for a, b in zip(baseline_cases, selected_cases, strict=True)),
        "became_exact": sum(not a["exact"] and b["exact"] for a, b in zip(baseline_cases, selected_cases, strict=True)),
        "lost_exactness": sum(a["exact"] and not b["exact"] for a, b in zip(baseline_cases, selected_cases, strict=True)),
    }
    report = {
        "source_files_sha256": source_hashes(),
        "experiment": "shape_cnn_v1_threshold_sweep", "dataset": "shape_edits_v1",
        "split": "validation", "images": len(records), "threshold_grid": list(THRESHOLDS),
        "selection_rule": SELECTION_RULE, "selected_threshold": selected_threshold,
        "candidates": candidates, "paired_changes_vs_0_50": paired,
        "checkpoint_sha256": checkpoint_before,
        "manifest_sha256": reference["manifest_sha256"],
        "evaluated_files_sha256": evaluated_files.hexdigest(),
        "reference_report_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "comparison_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": "cpu", "torch_version": str(torch.__version__),
        "training_performed": False, "test_dataset_used": False,
        "score_interpretation": "Sigmoid scores are not demonstrated to be calibrated probabilities of correctness.",
        "scope": "Validation-selected operating point on one synthetic dataset and frozen checkpoint; test performance is not yet established.",
    }
    configuration = {
        "experiment": report["experiment"], "model": "ShapeEditor",
        "dataset": "shape_edits_v1", "checkpoint_sha256": checkpoint_before,
        "manifest_sha256": reference["manifest_sha256"],
        "mask_probability_threshold": selected_threshold, "renderer": reference["renderer"],
        "selected_on": "validation", "selection_rule": SELECTION_RULE,
        "test_dataset_used": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "selected_config.json").write_text(json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "cases.jsonl").write_text("".join(json.dumps(case, sort_keys=True) + "\n" for cases in per_case.values() for case in cases), encoding="utf-8")
    worst = sorted(range(len(records)), key=lambda index: (-baseline_cases[index]["wrong_pixels"], records[index]["id"]))[:4]
    save_preview(worst, records, images, logits, selected_threshold, output / "preview.png")
    for candidate in candidates:
        scores = candidate["scores"]
        counts = scores["counts"]
        print(f"Threshold {candidate['threshold']:.2f}: exact images={counts['exact_images']}/{counts['images']}; circle edit={scores['edit_pixel_accuracy']:.3%}; square preserved={scores['square_preservation']:.3%}; wrong pixels={counts['pixels'] - counts['correct_pixels']}")
    print(f"Selected threshold: {selected_threshold:.2f}. {SELECTION_RULE}")
    print(f"Paired changes: {json.dumps(paired, sort_keys=True)}")
    print(f"Report, selected configuration, and preview: {output}")


if __name__ == "__main__":
    main()
