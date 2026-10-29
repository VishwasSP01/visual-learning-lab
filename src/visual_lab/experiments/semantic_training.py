"""Learn a shared three-class scene representation, then select by instruction."""

import argparse
import hashlib
import io
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from visual_lab.data.loaders import load_instruction_split
from visual_lab.experiments.instruction_training import evaluate_logits, predict_logits, scene_indices, state_digest
from visual_lab.experiments.instruction_thresholds import error_breakdown, paired_changes
from visual_lab.models import InstructionEditor, SemanticDilatedEditor, SemanticLiteUNet
from visual_lab.paths import ROOT
from visual_lab.provenance import run_metadata
from visual_lab.rendering import render_edit
from visual_lab.thresholds import apply_threshold


MODELS = {"dilated": SemanticDilatedEditor, "lite_unet": SemanticLiteUNet}


def collapse_scenes(data):
    """Derive supervised semantic labels from paired masks, once per scene."""
    images, labels, mapping = [], [], np.empty(len(data[3]), dtype=np.int64)
    for scene_index, pair in enumerate(scene_indices(data[3]).values()):
        circle, square = pair[0], pair[1]
        if not torch.equal(data[0][circle], data[0][square]):
            raise ValueError("Paired requests must have identical RGB inputs.")
        if (data[2][circle].bool() & data[2][square].bool()).any():
            raise ValueError("Semantic supervision masks must be disjoint.")
        images.append(data[0][circle])
        labels.append(data[2][circle] + 2 * data[2][square])
        mapping[circle] = mapping[square] = scene_index
    return torch.stack(images), torch.stack(labels), mapping


def segmentation_loss(logits, labels, class_weights, dice_weight):
    # Fixed training-derived weights; a pixel mean allows exact batch aggregation.
    ce = F.cross_entropy(logits, labels, weight=class_weights, reduction="none").mean()
    if dice_weight == 0:
        return ce
    probabilities = logits.softmax(dim=1)[:, 1:]
    foreground = F.one_hot(labels, num_classes=3).permute(0, 3, 1, 2)[:, 1:].to(logits.dtype)
    intersection = (probabilities * foreground).sum(dim=(2, 3))
    denominator = probabilities.sum(dim=(2, 3)) + foreground.sum(dim=(2, 3))
    dice = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return ce + dice_weight * dice


def loader(data, batch_size, seed, shuffle=False):
    return DataLoader(TensorDataset(*data[:2]), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def prepare(images, labels, device):
    return (images.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).contiguous() / 255,
            labels.to(device=device, dtype=torch.long))


@torch.no_grad()
def validation_loss(model, batches, weights, dice_weight, device):
    model.eval()
    total, scenes = 0.0, 0
    for images, labels in batches:
        images, labels = prepare(images, labels, device)
        total += segmentation_loss(model(images), labels, weights, dice_weight).item() * len(images)
        scenes += len(images)
    return total / scenes


@torch.inference_mode()
def predict_scene_classes(model, images, batch_size=64):
    """RGB only. Argmax gives mutually exclusive background/circle/square classes."""
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].permute(0, 3, 1, 2).float().contiguous() / 255
        logits = model(batch)
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite semantic logits.")
        outputs.append(logits.argmax(dim=1).numpy().astype(np.uint8))
    return np.concatenate(outputs)


def request_logits(classes, records, mapping):
    """Select the learned class with the categorical ID; no label masks involved."""
    task_classes = np.array([r["instruction_id"] + 1 for r in records])
    masks = classes[mapping] == task_classes[:, None, None]
    # The common evaluator thresholds at 0.50 (logit 0). These are hard decisions,
    # not confidence scores, and are never used for a confidence-threshold sweep.
    return np.where(masks, 1.0, -1.0).astype(np.float32)


def convolution_macs(model, *inputs):
    """Count convolution multiply-accumulates; excludes bias/activation/resize work."""
    total = 0
    def count(module, args, output):
        nonlocal total
        total += output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
    hooks = [module.register_forward_hook(count) for module in model.modules() if isinstance(module, nn.Conv2d)]
    try:
        with torch.inference_mode():
            model(*inputs)
    finally:
        for hook in hooks:
            hook.remove()
    return total


@torch.inference_mode()
def benchmark(model, reference, image, threshold, warmups=10, repeats=100, decoder="argmax"):
    """Interleave CPU timings; normalized inputs ready, both models in eval mode."""
    ids = torch.tensor([0, 1])
    paired_images = image.repeat(2, 1, 1, 1)
    def semantic(single):
        if decoder == "comparisons":
            from visual_lab.semantic_decoding import semantic_object_masks
            masks = semantic_object_masks(model(image))
            return masks[:, 0] if single else masks.permute(1, 0, 2, 3)
        classes = model(image).argmax(dim=1)
        return classes == 1 if single else torch.stack((classes == 1, classes == 2))
    def baseline(single):
        logits = reference(image if single else paired_images, ids[:1] if single else ids)
        return logits >= math.log(threshold / (1 - threshold))
    operations = {"semantic_single_request": lambda: semantic(True),
                  "reference_single_request": lambda: baseline(True),
                  "semantic_two_requests_same_scene": lambda: semantic(False),
                  "reference_two_requests_same_scene": lambda: baseline(False)}
    for _ in range(warmups):
        for operation in operations.values():
            operation()
    samples = {key: [] for key in operations}
    for iteration in range(repeats):
        order = list(operations) if iteration % 2 == 0 else list(reversed(operations))
        for key in order:
            started = time.perf_counter_ns()
            operations[key]()
            samples[key].append((time.perf_counter_ns() - started) / 1e6)
    return {
        "device": "cpu", "threads": torch.get_num_threads(), "machine": platform.machine(),
        "platform": platform.platform(), "warmups_per_operation": warmups, "repeats_per_operation": repeats,
        "semantic_decoder": decoder,
        "timings_ms": {key: {"median": float(np.median(values)), "p90": float(np.percentile(values, 90))}
                       for key, values in samples.items()},
        "scope": "Model plus class/threshold decoding on normalized 64x64 tensors; excludes file I/O, input normalization, RGB rendering. Pair reuses one semantic pass; reference batches both requests.",
        "convolution_macs_single_request": {"semantic": convolution_macs(model, image),
                                             "reference": convolution_macs(reference, image, ids[:1])},
        "convolution_macs_two_requests_same_scene": {"semantic": convolution_macs(model, image),
                                                      "reference": convolution_macs(reference, paired_images, ids)},
        "mac_scope": "Convolution multiply-accumulates only; one MAC is not one FLOP. Resizing and other operators are excluded.",
    }


def save_preview(data_dir, data, logits, reference_logits, threshold, indices, output):
    cell, height = 128, 152
    sheet = Image.new("RGB", (4 * cell, 24 + len(indices) * height), "#e8e8e8")
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(("Input", "Previous model", "Semantic model", "Target")):
        draw.text((column * cell + 4, 6), title, fill="black")
    for row, i in enumerate(indices):
        record = data[3][i]
        original = Image.fromarray(data[0][i].numpy())
        with Image.open(data_dir / record["target"]) as image:
            target = image.copy()
        panels = (original, render_edit(original, apply_threshold(reference_logits[i], threshold)),
                  render_edit(original, logits[i] >= 0), target)
        for column, panel in enumerate(panels):
            sheet.paste(panel.resize((cell, cell), Image.Resampling.NEAREST), (column * cell, 24 + row * height))
        draw.text((4, 24 + row * height + cell + 4),
                  f"{record['scene_id']} | edit {record['requested_object']}", fill="black")
    sheet.save(output)


def main(argv=None, *, run_context=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(MODELS), required=True)
    parser.add_argument("--dice-weight", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/instruction_edits_v1")
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.epochs <= 0 or args.batch_size <= 0 or not math.isfinite(args.learning_rate)
            or args.learning_rate <= 0 or not 0 <= args.seed < 2**32 or not 0 <= args.dice_weight <= 1):
        parser.error("Invalid training hyperparameters.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output must be empty; prior results are preserved.")
    metadata = run_metadata(args, seed=args.seed, configuration=run_context)
    manifest = (args.data_dir / "manifest.jsonl").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != args.manifest_sha256:
        parser.error("Manifest differs from the declared SHA-256.")
    reference_config_bytes = args.reference_config.read_bytes()
    reference_config = json.loads(reference_config_bytes)
    if (reference_config["model"] != "InstructionEditor" or reference_config["selected_on"] != "validation"
            or reference_config["manifest_sha256"] != args.manifest_sha256
            or not 0 < reference_config["mask_probability_threshold"] < 1):
        raise ValueError("Expected a validation-selected reference on the same dataset.")
    reference_report_bytes = (args.reference_run_dir / "validation.json").read_bytes()
    reference_report = json.loads(reference_report_bytes)
    if (reference_report["manifest_sha256"] != args.manifest_sha256
            or reference_report["split"] != "validation"
            or reference_report["checkpoint_sha256"]["conditioned"] != reference_config["checkpoint_sha256"]):
        raise ValueError("Reference run and selected configuration disagree.")
    reference_bytes = (args.reference_run_dir / "conditioned.pt").read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != reference_config["checkpoint_sha256"]:
        raise ValueError("Reference checkpoint differs from the selected configuration.")
    reference = InstructionEditor().cpu().eval()
    reference.load_state_dict(torch.load(io.BytesIO(reference_bytes), map_location="cpu", weights_only=True)["model_state"], strict=True)
    records = [json.loads(line) for line in manifest.decode().splitlines()]
    train_requests = load_instruction_split(args.data_dir, records, "train")
    val_requests = load_instruction_split(args.data_dir, records, "validation")
    if val_requests[4] != reference_report["validation_input_mask_sha256"]:
        raise ValueError("Validation input/mask files differ from the reference run.")
    train, validation = collapse_scenes(train_requests), collapse_scenes(val_requests)
    name = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    name = "cpu" if name == "auto" else name
    if name == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable; use CPU.")
    device = torch.device(name)
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model = MODELS[args.architecture]().to(device)
    initial_hash = state_digest(model.state_dict())
    class_counts = torch.bincount(train[1].flatten().long(), minlength=3)
    if (class_counts == 0).any():
        raise ValueError("Every semantic class must occur in training.")
    weights = (class_counts.sum() / (3 * class_counts.float())).to(device)
    train_loader = loader(train, args.batch_size, args.seed, shuffle=True)
    val_loader = loader(validation, args.batch_size, args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss, best_state, best_epoch, history = float("inf"), None, None, []
    started = time.perf_counter()
    print(f"Architecture={args.architecture}; Dice weight={args.dice_weight}; device={device}; train scenes={len(train[0])}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for images, labels in train_loader:
            images, labels = prepare(images, labels, device)
            loss = segmentation_loss(model(images), labels, weights, args.dice_weight)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss.")
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(images)
        val_loss = validation_loss(model, val_loader, weights, args.dice_weight, device)
        if not math.isfinite(val_loss):
            raise RuntimeError("Non-finite validation loss.")
        history.append({"epoch": epoch, "training_loss": total / len(train[0]), "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}: train loss={total / len(train[0]):.6f}; validation loss={val_loss:.6f}", flush=True)
    training_seconds = time.perf_counter() - started
    model = MODELS[args.architecture]().cpu().eval()
    model.load_state_dict(best_state, strict=True)
    classes = predict_scene_classes(model, validation[0], args.batch_size)
    logits = request_logits(classes, val_requests[3], validation[2])
    reference_logits = predict_logits(reference, val_requests, args.batch_size)
    threshold = reference_config["mask_probability_threshold"]
    scores, cases = evaluate_logits(args.data_dir, val_requests, logits, 0.5, "semantic")
    reference_scores, reference_cases = evaluate_logits(args.data_dir, val_requests, reference_logits, threshold, "reference")
    expected_labels = reference_report["methods"]["conditioned"]["target_preservation_files_sha256"]
    if any(s["target_preservation_files_sha256"] != expected_labels for s in (scores, reference_scores)):
        raise ValueError("Evaluated labels changed between methods.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "model.pt"
    torch.save({"model_state": best_state, "architecture": args.architecture, "seed": args.seed,
                "selected_epoch": best_epoch, "manifest_sha256": args.manifest_sha256,
                "class_mapping": {"background": 0, "circle": 1, "square": 2}}, checkpoint_path)
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    worst = sorted(range(len(cases)), key=lambda i: (-cases[i]["wrong_pixels"], cases[i]["id"]))[:4]
    first = list(range(min(4, len(cases))))
    save_preview(args.data_dir, val_requests, logits, reference_logits, threshold, first, args.output_dir / "preview.png")
    save_preview(args.data_dir, val_requests, logits, reference_logits, threshold, worst, args.output_dir / "worst_cases.png")
    efficiency = benchmark(model, reference, validation[0][:1].permute(0, 3, 1, 2).float() / 255, threshold)
    report = {
        **metadata, "experiment": run_context["declared"]["experiment"] if run_context else "semantic_editor",
        "dataset": "instruction_edits_v1", "split": "validation", "architecture": args.architecture,
        "parameters": sum(p.numel() for p in model.parameters()),
        "reference_parameters": sum(p.numel() for p in reference.parameters()),
        "manifest_sha256": args.manifest_sha256, "checkpoint_sha256": checkpoint_sha,
        "reference_checkpoint_sha256": reference_config["checkpoint_sha256"],
        "reference_config_sha256": hashlib.sha256(reference_config_bytes).hexdigest(),
        "reference_report_sha256": hashlib.sha256(reference_report_bytes).hexdigest(),
        "reference_threshold": threshold, "reference_checkpoint_bytes": len(reference_bytes),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "training_input_mask_sha256": train_requests[4], "validation_input_mask_sha256": val_requests[4],
        "train_scenes": len(train[0]), "validation_scenes": len(validation[0]), "validation_requests": len(cases),
        "loss": "Training-frequency-balanced pixel-mean cross entropy + dice_weight * mean per-image foreground soft Dice loss (linear denominator, smooth=1).",
        "dice_weight": args.dice_weight, "training_class_counts": class_counts.tolist(),
        "class_weights": weights.cpu().tolist(), "class_mapping": {"background": 0, "circle": 1, "square": 2},
        "decoder": "Argmax over three predicted classes; instruction ID+1 selects that class; fixed red renderer. No threshold search.",
        "selected_epoch": best_epoch, "selected_validation_loss": best_loss,
        "epochs": args.epochs, "batch_size": args.batch_size, "optimizer": "Adam", "learning_rate": args.learning_rate,
        "selection": "Lowest validation training-objective loss across the declared epochs; ties keep earliest epoch.",
        "initial_state_sha256": initial_hash, "selected_state_sha256": state_digest(best_state), "history": history,
        "training_device": str(device), "evaluation_device": "cpu", "torch_version": str(torch.__version__),
        "training_seconds": training_seconds, "efficiency": efficiency,
        "methods": {"reference": reference_scores, "semantic": scores},
        "errors": error_breakdown(scores),
        "errors_per_instruction": {key: error_breakdown(value) for key, value in scores["per_instruction"].items()},
        "paired_changes_vs_reference": paired_changes(reference_cases, cases),
        "preview_request_ids": [cases[i]["id"] for i in first],
        "worst_request_ids": [cases[i]["id"] for i in worst],
        "test_images_loaded": False, "test_model_performance_measured": False,
        "scope": "Specialized three-class perception plus categorical routing on synthetic scenes. No learned language understanding, real-photo claim, or clean architecture-only comparison with the historical binary model; supervision/objective and budget differ.",
    }
    selected_config = {"model": type(model).__name__, "architecture": args.architecture,
                       "dataset": report["dataset"], "manifest_sha256": args.manifest_sha256,
                       "checkpoint_sha256": checkpoint_sha, "decoder": "three_class_argmax",
                       "class_mapping": report["class_mapping"], "selected_on": "validation",
                       "test_model_performance_measured": False}
    for file, value in (("validation.json", report), ("selected_config.json", selected_config)):
        (args.output_dir / file).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "cases.jsonl").write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in reference_cases + cases))
    print(f"Selected epoch={best_epoch}; exact requests={scores['counts']['exact_images']}/{len(cases)}; "
          f"exact pairs={scores['scene_counts']['exact_scenes']}/{len(validation[0])}; errors={report['errors']}", flush=True)
    print(f"Efficiency: {json.dumps(efficiency, sort_keys=True)}", flush=True)
    print(f"Report: {args.output_dir / 'validation.json'}", flush=True)
