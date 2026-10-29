"""Evaluate one committed semantic editor and reference on reserved test scenes."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from visual_lab.data.instruction_pairs import audit_records
from visual_lab.data.loaders import load_spatial_split
from visual_lab.experiments.instruction_training import evaluate_logits, predict_logits, scene_indices
from visual_lab.experiments.instruction_thresholds import error_breakdown, paired_changes
from visual_lab.experiments.semantic_training import predict_scene_classes, request_logits, save_preview
from visual_lab.models import InstructionEditor, SemanticDilatedEditor
from visual_lab.provenance import run_metadata
from visual_lab.semantic_artifacts import validate_semantic_configuration
from visual_lab.thresholds import apply_threshold


def validate_frozen_plan(plan, weights, reference_weights, manifest):
    if (plan.get("schema_version") != 1 or plan.get("split") != "test"
            or plan.get("selection_performed") is not False
            or plan.get("frozen_before_first_test_performance") is not True):
        raise ValueError("Expected a frozen, pre-test evaluation plan with no selection.")
    count = plan.get("expected_scenes")
    if type(count) is not int or count <= 0 or plan.get("expected_requests") != 2 * count:
        raise ValueError("Expected one pair of requests per test scene.")
    validate_semantic_configuration(plan["semantic"], weights, manifest)
    reference = plan["reference"]
    if (reference.get("model") != "InstructionEditor" or reference.get("dataset") != "instruction_edits_v1"
            or reference.get("selected_on") != "validation"
            or reference.get("test_model_performance_measured") is not False
            or reference.get("manifest_sha256") != plan["semantic"]["manifest_sha256"]):
        raise ValueError("Reference must be validation-selected on the same dataset.")
    if hashlib.sha256(reference_weights).hexdigest() != reference["checkpoint_sha256"]:
        raise ValueError("Reference checkpoint differs from the frozen plan.")
    apply_threshold(np.array([0.0]), reference["mask_probability_threshold"])


def load_test(data_dir, manifest, plan):
    """Dedicated frozen-evaluation entry; training's test prohibition stays intact."""
    records = [json.loads(line) for line in manifest.decode().splitlines()]
    audit_records(records)  # Manifest only; no train/validation image is read.
    selected = [r for r in records if r["split"] == "test"]
    if len(selected) != plan["expected_requests"] or len(scene_indices(selected)) != plan["expected_scenes"]:
        raise ValueError("Test split size differs from the committed frozen plan.")
    inputs, labels, selected, digest = load_spatial_split(data_dir, records, "test")
    for record, pixels in zip(selected, inputs, strict=True):
        if hashlib.sha256(pixels.numpy().tobytes()).hexdigest() != record["input_sha256"]:
            raise ValueError("Test input pixels differ from the manifest.")
    ids = torch.tensor([r["instruction_id"] for r in selected], dtype=torch.long)
    return inputs, ids, labels, selected, digest


def main(argv=None, *, run_context=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "run-dir", "reference-run-dir", "frozen-config", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output must be empty; the recorded test evaluation is preserved.")
    metadata = run_metadata(args, configuration=run_context)
    plan_bytes = args.frozen_config.read_bytes()
    plan = json.loads(plan_bytes)
    manifest_path = args.data_dir / "manifest.jsonl"
    manifest = manifest_path.read_bytes()
    checkpoint_path = args.run_dir / "model.pt"
    reference_path = args.reference_run_dir / "conditioned.pt"
    weights, reference_weights = checkpoint_path.read_bytes(), reference_path.read_bytes()
    validate_frozen_plan(plan, weights, reference_weights, manifest)
    model = SemanticDilatedEditor().cpu().eval()
    model.load_state_dict(torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True)["model_state"], strict=True)
    reference = InstructionEditor().cpu().eval()
    reference.load_state_dict(torch.load(io.BytesIO(reference_weights), map_location="cpu", weights_only=True)["model_state"], strict=True)
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_started.json").write_text(json.dumps({
        "frozen_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), **metadata,
    }, indent=2, sort_keys=True) + "\n")
    data = load_test(args.data_dir, manifest, plan)
    pairs = scene_indices(data[3])
    scene_images, mapping = [], np.empty(len(data[3]), dtype=np.int64)
    for scene, pair in enumerate(pairs.values()):
        scene_images.append(data[0][pair[0]])
        mapping[pair[0]] = mapping[pair[1]] = scene
    classes = predict_scene_classes(model, torch.stack(scene_images), batch_size=64)
    logits = request_logits(classes, data[3], mapping)
    reference_logits = predict_logits(reference, data, batch_size=64)
    threshold = plan["reference"]["mask_probability_threshold"]
    scores, cases = evaluate_logits(args.data_dir, data, logits, .5, "frozen_semantic")
    reference_scores, reference_cases = evaluate_logits(args.data_dir, data, reference_logits, threshold, "frozen_reference")
    if scores["target_preservation_files_sha256"] != reference_scores["target_preservation_files_sha256"]:
        raise ValueError("Test labels changed between methods.")
    predictions_path = args.output_dir / "predicted_masks.npz"
    np.savez_compressed(predictions_path, semantic=logits >= 0,
                        reference=apply_threshold(reference_logits, threshold))
    if (checkpoint_path.read_bytes() != weights or reference_path.read_bytes() != reference_weights
            or args.frozen_config.read_bytes() != plan_bytes or manifest_path.read_bytes() != manifest):
        raise ValueError("A frozen model, plan, or manifest changed during evaluation.")
    save_preview(args.data_dir, data, logits, reference_logits, threshold,
                 list(range(min(4, len(data[3])))), args.output_dir / "preview.png")
    report = {
        **metadata, "experiment": run_context["declared"]["experiment"] if run_context else "semantic_dilated_v2_frozen_test",
        "dataset": "instruction_edits_v1", "split": "test", "scenes": len(pairs), "requests": len(data[3]),
        "frozen_plan": plan, "frozen_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "checkpoint_sha256": plan["semantic"]["checkpoint_sha256"],
        "reference_checkpoint_sha256": plan["reference"]["checkpoint_sha256"],
        "manifest_sha256": plan["semantic"]["manifest_sha256"], "test_input_mask_sha256": data[4],
        "saved_predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "methods": {"frozen_reference": reference_scores, "frozen_semantic": scores},
        "errors": error_breakdown(scores),
        "errors_per_instruction": {name: error_breakdown(task) for name, task in scores["per_instruction"].items()},
        "paired_changes_vs_reference": paired_changes(reference_cases, cases),
        "preview_request_ids": [r["id"] for r in data[3][:4]],
        "preview_selection": "First four test requests in manifest order; no outcome selection.",
        "device": "cpu", "torch_version": str(torch.__version__), "training_performed": False,
        "selection_performed": False, "threshold_selection_performed": False,
        "scope": "One frozen model/reference comparison on previously reserved synthetic test scenes. Construction integrity checks read test files earlier; model performance did not guide selection. This test is now observed.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "cases.jsonl").write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in reference_cases + cases))
    for name, method in report["methods"].items():
        print(f"{name}: exact requests={method['counts']['exact_images']}/{len(data[3])}; "
              f"exact pairs={method['scene_counts']['exact_scenes']}/{len(pairs)}", flush=True)
    print(f"Errors: {report['errors']}; report: {args.output_dir / 'report.json'}", flush=True)
