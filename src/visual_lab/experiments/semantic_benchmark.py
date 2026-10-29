"""Verify an equivalent semantic decoder and benchmark fixed checkpoints on CPU."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from visual_lab.data.loaders import load_instruction_split
from visual_lab.experiments.semantic_training import MODELS, benchmark, collapse_scenes
from visual_lab.models import InstructionEditor
from visual_lab.provenance import run_metadata
from visual_lab.semantic_decoding import semantic_object_masks


def verified_state(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Checkpoint differs from selected configuration: {path}")
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)["model_state"]


def main(argv=None, *, run_context=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "run-dir", "reference-run-dir", "reference-config", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output must be empty; previous results are preserved.")
    metadata = run_metadata(args, configuration=run_context)
    config_bytes = (args.run_dir / "selected_config.json").read_bytes()
    config = json.loads(config_bytes)
    reference_bytes = args.reference_config.read_bytes()
    reference_config = json.loads(reference_bytes)
    if (config["selected_on"] != "validation" or reference_config["selected_on"] != "validation"
            or config["manifest_sha256"] != reference_config["manifest_sha256"]):
        raise ValueError("Expected validation-selected models on the same manifest.")
    model = MODELS[config["architecture"]]().eval()
    model.load_state_dict(verified_state(args.run_dir / "model.pt", config["checkpoint_sha256"]), strict=True)
    reference = InstructionEditor().eval()
    reference.load_state_dict(verified_state(args.reference_run_dir / "conditioned.pt", reference_config["checkpoint_sha256"]), strict=True)
    manifest = (args.data_dir / "manifest.jsonl").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Manifest differs from selected configurations.")
    records = [json.loads(line) for line in manifest.decode().splitlines()]
    data = load_instruction_split(args.data_dir, records, "validation")
    scenes = collapse_scenes(data)
    torch.set_num_threads(4)
    with torch.inference_mode():
        for start in range(0, len(scenes[0]), 37):
            inputs = scenes[0][start:start+37].permute(0, 3, 1, 2).float().contiguous() / 255
            logits = model(inputs)
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite semantic logits.")
            classes = logits.argmax(dim=1)
            expected = torch.stack((classes == 1, classes == 2), dim=1)
            if not torch.equal(semantic_object_masks(logits), expected):
                raise ValueError("Optimized decoder changed a predicted edit mask.")
    image = scenes[0][:1].permute(0, 3, 1, 2).float() / 255
    timings = {decoder: benchmark(model, reference, image, reference_config["mask_probability_threshold"], decoder=decoder)
               for decoder in ("argmax", "comparisons")}
    report = {
        **metadata, "experiment": run_context["declared"]["experiment"] if run_context else "semantic_decoder_benchmark",
        "checkpoint_sha256": config["checkpoint_sha256"], "reference_checkpoint_sha256": reference_config["checkpoint_sha256"],
        "selected_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "reference_config_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "manifest_sha256": config["manifest_sha256"], "validation_input_mask_sha256": data[4],
        "equivalent_scene_predictions": len(scenes[0]), "equivalent_request_masks": len(data[3]),
        "timings": timings, "torch_version": str(torch.__version__),
        "training_performed": False, "accuracy_selection_performed": False, "test_images_loaded": False,
        "scope": "Prediction-preserving decoder optimization and local CPU timing. Finite logits are checked for all validation scenes before timing; the timed fast path assumes finite outputs.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"equivalent_request_masks": len(data[3]), "timings": timings}, indent=2), flush=True)
