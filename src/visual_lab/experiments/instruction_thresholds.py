"""Select one shared instruction-model threshold using validation scene pairs."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from visual_lab.data.loaders import load_instruction_split
from visual_lab.experiments.instruction_training import evaluate_logits, predict_logits, scene_indices
from visual_lab.models import InstructionEditor
from visual_lab.paths import ROOT
from visual_lab.provenance import run_metadata
from visual_lab.rendering import render_edit
from visual_lab.thresholds import THRESHOLDS, apply_threshold


SELECTION_RULE = (
    "Most exact validation scene pairs; then most exact requests; "
    "then fewest wrong pixels; then lowest threshold."
)


def select_candidate(candidates):
    if not candidates:
        raise ValueError("No threshold candidates.")
    return max(candidates, key=lambda c: (
        c["scores"]["scene_counts"]["exact_scenes"],
        c["scores"]["counts"]["exact_images"],
        -(c["scores"]["counts"]["pixels"] - c["scores"]["counts"]["correct_pixels"]),
        -c["threshold"],
    ))


def error_breakdown(scores):
    c = scores["counts"]
    return {
        "missed_requested_pixels": c["edited_pixels"] - c["correct_edited_pixels"],
        "damaged_opposite_pixels": c["preserved_object_pixels"] - c["unchanged_object_pixels"],
        "damaged_background_pixels": c["white_background_pixels"] - c["unchanged_white_background_pixels"],
        "wrong_pixels": c["pixels"] - c["correct_pixels"],
    }


def paired_changes(before, after):
    pairs = list(zip(before, after, strict=True))
    if any((a["id"], a["scene_id"], a["instruction_id"]) !=
           (b["id"], b["scene_id"], b["instruction_id"]) for a, b in pairs):
        raise ValueError("Request order changed across thresholds.")
    groups = scene_indices(before)
    return {
        "requests_with_fewer_wrong_pixels": sum(b["wrong_pixels"] < a["wrong_pixels"] for a, b in pairs),
        "requests_with_more_wrong_pixels": sum(b["wrong_pixels"] > a["wrong_pixels"] for a, b in pairs),
        "requests_with_equal_wrong_pixels": sum(b["wrong_pixels"] == a["wrong_pixels"] for a, b in pairs),
        "requests_became_exact": sum(not a["exact"] and b["exact"] for a, b in pairs),
        "requests_lost_exactness": sum(a["exact"] and not b["exact"] for a, b in pairs),
        "scene_pairs_became_exact": sum(
            not all(before[i]["exact"] for i in pair.values()) and
            all(after[i]["exact"] for i in pair.values()) for pair in groups.values()),
        "scene_pairs_lost_exactness": sum(
            all(before[i]["exact"] for i in pair.values()) and
            not all(after[i]["exact"] for i in pair.values()) for pair in groups.values()),
    }


def save_preview(data_dir, data, logits, selected, cases, output):
    groups = scene_indices(data[3])
    worst = sorted(groups, key=lambda scene: (
        -sum(cases[i]["wrong_pixels"] for i in groups[scene].values()), scene))[:2]
    indices = [groups[scene][task] for scene in worst for task in (0, 1)]
    cell, row_height = 128, 152
    sheet = Image.new("RGB", (4 * cell, 24 + len(indices) * row_height), "#e8e8e8")
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(("Input", "Threshold 0.50", f"Selected {selected:.2f}", "Target")):
        draw.text((column * cell + 4, 6), title, fill="black")
    for row, index in enumerate(indices):
        record = data[3][index]
        original = Image.fromarray(data[0][index].numpy())
        with Image.open(data_dir / record["target"]) as image:
            target = image.copy()
        panels = (original, render_edit(original, apply_threshold(logits[index], 0.5)),
                  render_edit(original, apply_threshold(logits[index], selected)), target)
        for column, panel in enumerate(panels):
            sheet.paste(panel.resize((cell, cell), Image.Resampling.NEAREST),
                        (column * cell, 24 + row * row_height))
        draw.text((4, 24 + row * row_height + cell + 4),
                  f"{record['scene_id']} | edit {record['requested_object']}", fill="black")
    sheet.save(output)
    return [data[3][i]["id"] for i in indices]


def main(argv=None, *, run_context=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/instruction_edits_v1")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/reproductions/instruction_cnn_v2")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.resolve() == args.run_dir.resolve():
        parser.error("Output must be separate from the reference run.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty; previous results are preserved.")
    metadata = run_metadata(args, configuration=run_context)
    reference_bytes = (args.run_dir / "validation.json").read_bytes()
    reference = json.loads(reference_bytes)
    if (reference["dataset"] != "instruction_edits_v1" or reference["split"] != "validation"
            or reference["mask_probability_threshold"] != 0.5):
        raise ValueError("Expected an instruction validation reference at threshold 0.50.")
    manifest = (args.data_dir / "manifest.jsonl").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != reference["manifest_sha256"]:
        raise ValueError("Manifest differs from the reference run.")
    checkpoint_path = args.run_dir / "conditioned.pt"
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    if checkpoint_sha != reference["checkpoint_sha256"]["conditioned"]:
        raise ValueError("Conditioned checkpoint differs from the reference run.")
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    if checkpoint["manifest_sha256"] != reference["manifest_sha256"]:
        raise ValueError("Checkpoint manifest mismatch.")
    torch.set_num_threads(4)
    model = InstructionEditor().cpu().eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    records = [json.loads(line) for line in manifest.decode().splitlines()]
    data = load_instruction_split(args.data_dir, records, "validation")
    if data[4] != reference["validation_input_mask_sha256"]:
        raise ValueError("Validation input/mask files changed.")
    logits = predict_logits(model, data, batch_size=64)
    candidates, cases_by_threshold, previous = [], {}, None
    for threshold in THRESHOLDS:
        masks = apply_threshold(logits, threshold)
        if previous is not None and (masks & ~previous).any():
            raise ValueError("Increasing the threshold introduced edit pixels.")
        previous = masks
        scores, cases = evaluate_logits(args.data_dir, data, logits, threshold, "conditioned")
        if scores["target_preservation_files_sha256"] != reference["methods"]["conditioned"]["target_preservation_files_sha256"]:
            raise ValueError("Validation target/preservation files changed.")
        candidates.append({"threshold": threshold, "scores": scores,
                           "errors": error_breakdown(scores),
                           "errors_per_instruction": {name: error_breakdown(task)
                                                      for name, task in scores["per_instruction"].items()}})
        cases_by_threshold[threshold] = [{**case, "threshold": threshold} for case in cases]
    # Reproduce more than the headline metric before accepting a sweep result.
    if candidates[0]["scores"] != reference["methods"]["conditioned"]:
        raise ValueError("Threshold 0.50 does not reproduce the reference validation scores.")
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_sha:
        raise ValueError("Checkpoint changed during the sweep.")
    selected = select_candidate(candidates)
    threshold = selected["threshold"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_ids = save_preview(args.data_dir, data, logits, threshold, cases_by_threshold[0.5],
                               args.output_dir / "preview.png")
    report = {
        **metadata,
        "experiment": run_context["declared"]["experiment"] if run_context else "instruction_cnn_v2_thresholds",
        "dataset": "instruction_edits_v1", "split": "validation",
        "threshold_grid": list(THRESHOLDS), "selection_rule": SELECTION_RULE,
        "selected_threshold": threshold, "candidates": candidates,
        "paired_changes_vs_0_50": paired_changes(cases_by_threshold[0.5], cases_by_threshold[threshold]),
        "checkpoint_sha256": checkpoint_sha, "manifest_sha256": reference["manifest_sha256"],
        "validation_input_mask_sha256": data[4],
        "reference_report_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "preview_request_ids": preview_ids,
        "preview_selection": "Both requests on the two scenes with most wrong pixels at 0.50; tie by scene ID.",
        "evaluation_device": "cpu", "torch_version": str(torch.__version__),
        "training_performed": False, "test_images_loaded": False, "test_model_performance_measured": False,
        "scope": "Validation-selected shared cutoff on the frozen categorical instruction model; no new test result.",
        "score_interpretation": "Sigmoid outputs are not established as calibrated correctness probabilities.",
    }
    configuration = {
        "model": "InstructionEditor", "dataset": report["dataset"],
        "checkpoint_sha256": checkpoint_sha, "manifest_sha256": reference["manifest_sha256"],
        "mask_probability_threshold": threshold, "selected_on": "validation",
        "selection_rule": SELECTION_RULE, "threshold_grid": list(THRESHOLDS),
        "test_model_performance_measured": False,
    }
    for name, value in (("report.json", report), ("selected_config.json", configuration)):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "cases.jsonl").write_text("".join(
        json.dumps(case, sort_keys=True) + "\n" for cases in cases_by_threshold.values() for case in cases))
    for candidate in candidates:
        scores = candidate["scores"]
        print(f"Threshold {candidate['threshold']:.2f}: exact requests={scores['counts']['exact_images']}; "
              f"exact scene pairs={scores['scene_counts']['exact_scenes']}; errors={candidate['errors']}", flush=True)
    print(f"Selected threshold: {threshold:.2f}. {SELECTION_RULE}", flush=True)
    print(f"Report: {args.output_dir / 'report.json'}", flush=True)
