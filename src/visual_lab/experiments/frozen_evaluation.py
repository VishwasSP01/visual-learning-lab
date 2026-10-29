"""Evaluate a validation-selected, checkpoint-bound configuration on the test split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from visual_lab.baselines import recolor_blue
from visual_lab.metrics import spatial_counts, summarize_spatial
from visual_lab.models import ShapeEditor
from visual_lab.paths import ROOT
from visual_lab.provenance import source_hashes
from visual_lab.rendering import render_edit
from visual_lab.thresholds import apply_threshold


def validate_configuration(config, checkpoint_bytes, manifest_bytes):
    """Reject a different model/data pair before loading any test image."""
    if config.get("selected_on") != "validation" or config.get("test_dataset_used") is not False:
        raise ValueError("Configuration must have been selected without test data.")
    if config.get("dataset") != "shape_edits_v1" or config.get("model") != "ShapeEditor":
        raise ValueError("Unsupported model or dataset in the frozen configuration.")
    if hashlib.sha256(checkpoint_bytes).hexdigest() != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the validation-selected checkpoint.")
    if hashlib.sha256(manifest_bytes).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the frozen configuration.")
    threshold = config["mask_probability_threshold"]
    apply_threshold(np.array([0.0]), threshold)  # Check the declared probability range.
    return threshold


def save_preview(examples, threshold, output):
    cell, row_height = 128, 152
    sheet = Image.new("RGB", (4 * cell, 24 + len(examples) * row_height), "#e8e8e8")
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(("Input", "Color rule", f"Frozen CNN {threshold:.2f}", "Target")):
        draw.text((column * cell + 4, 6), label, fill="black")
    for row, (example_id, images) in enumerate(examples):
        for column, image in enumerate(images):
            sheet.paste(image.resize((cell, cell), Image.Resampling.NEAREST),
                        (column * cell, 24 + row * row_height))
        draw.text((5, 24 + row * row_height + cell + 4), example_id, fill="black")
    sheet.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/shape_edits_v1")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/spatial_cnn")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "docs/experiments/shape_cnn_v1_selected_config.json")
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    manifest = (args.data_dir / "manifest.jsonl").read_bytes()
    checkpoint_path = args.run_dir / "model.pt"
    checkpoint_bytes = checkpoint_path.read_bytes()
    threshold = validate_configuration(config, checkpoint_bytes, manifest)
    records = [json.loads(line) for line in manifest.decode().splitlines()]
    records = [record for record in records if record["split"] == "test"]
    if len(records) != 1000:
        raise ValueError("The frozen experiment expects exactly 1,000 test examples.")
    model = ShapeEditor()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model_state"])
    model.eval()
    torch.set_num_threads(4)
    totals = {"color_rule": {}, "frozen_spatial_cnn": {}}
    cases, examples = [], []
    evaluated_files = hashlib.sha256()
    for start in range(0, len(records), 32):
        batch = records[start:start + 32]
        loaded = []
        for record in batch:
            images = {}
            for kind in ("input", "target", "mask", "square_mask"):
                path = args.data_dir / record[kind]
                evaluated_files.update(record[kind].encode() + b"\0" + path.read_bytes())
                with Image.open(path) as image:
                    images[kind] = image.copy()
            loaded.append(images)
        inputs = torch.from_numpy(np.stack([np.array(images["input"]) for images in loaded]))
        inputs = inputs.permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            logits = model(inputs)[:, 0].numpy()
        if not np.isfinite(logits).all():
            raise ValueError("Non-finite model output.")
        predictions = apply_threshold(logits, threshold)
        for record, images, predicted_mask in zip(batch, loaded, predictions, strict=True):
            original, target = images["input"], images["target"]
            outputs = {"color_rule": recolor_blue(original),
                       "frozen_spatial_cnn": render_edit(original, predicted_mask)}
            for method, prediction in outputs.items():
                counts = spatial_counts(original, prediction, target, images["mask"], images["square_mask"])
                for key, value in counts.items():
                    totals[method][key] = totals[method].get(key, 0) + value
                cases.append({
                    "id": record["id"], "split": "test", "method": method,
                    "exact": bool(counts["exact_images"]),
                    "wrong_pixels": counts["pixels"] - counts["correct_pixels"],
                    "missed_circle_pixels": counts["edited_pixels"] - counts["correct_edited_pixels"],
                    "damaged_square_pixels": counts["square_pixels"] - counts["unchanged_square_pixels"],
                    "damaged_background_pixels": counts["white_background_pixels"] - counts["unchanged_white_background_pixels"],
                })
            if len(examples) < 4:
                examples.append((record["id"], (original, outputs["color_rule"], outputs["frozen_spatial_cnn"], target)))

    if checkpoint_path.read_bytes() != checkpoint_bytes or args.config.read_bytes() != config_bytes:
        raise ValueError("Frozen model or configuration changed during evaluation.")
    report = {
        "source_files_sha256": source_hashes(),
        "experiment": "shape_cnn_v1_frozen_test", "dataset": "shape_edits_v1",
        "split": "test", "images": len(records), "mask_probability_threshold": threshold,
        "checkpoint_sha256": config["checkpoint_sha256"],
        "manifest_sha256": config["manifest_sha256"],
        "frozen_configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "evaluated_files_sha256": evaluated_files.hexdigest(),
        "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": "cpu", "torch_version": str(torch.__version__),
        "selection_performed": False, "training_performed": False,
        "threshold_source": "Configuration selected on validation and committed before test evaluation.",
        "methods": {method: summarize_spatial(counts) for method, counts in totals.items()},
        "scope": "Held-out model-performance evaluation within a synthetic rendering process; no real-image or arbitrary-instruction claim.",
    }
    output = args.run_dir / "frozen_test"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "cases.jsonl").write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases), encoding="utf-8")
    save_preview(examples, threshold, output / "preview.png")
    for method, scores in report["methods"].items():
        print(f"{method}: exact images={scores['counts']['exact_images']}/{len(records)}; circle edit={scores['edit_pixel_accuracy']:.3%}; square preserved={scores['square_preservation']:.3%}; edit IoU={scores['edit_mask_iou']:.3%}")
    print(f"Frozen threshold {threshold:.2f}; no model or threshold selection performed.")
    print(f"Report and first-four-example preview: {output}")


if __name__ == "__main__":
    main()
