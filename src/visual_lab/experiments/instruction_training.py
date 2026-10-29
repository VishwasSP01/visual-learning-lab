"""Compare learned instruction conditioning with a matched constant-ID control."""

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from visual_lab.data.loaders import load_instruction_split
from visual_lab.instruction_metrics import instruction_counts, paired_scene_scores, summarize_requests
from visual_lab.models import InstructionEditor
from visual_lab.paths import ROOT
from visual_lab.provenance import run_metadata
from visual_lab.rendering import render_edit
from visual_lab.thresholds import apply_threshold


def make_loader(data, batch_size, seed, shuffle=False):
    return DataLoader(TensorDataset(*data[:3]), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def prepare(inputs, instruction_ids, masks, device, conditioned):
    images = inputs.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).contiguous() / 255
    ids = instruction_ids.to(device) if conditioned else torch.zeros_like(instruction_ids, device=device)
    labels = masks.unsqueeze(1).to(device=device, dtype=torch.float32)
    return images, ids, labels


@torch.no_grad()
def validation_loss(model, loader, criterion, device, conditioned):
    model.eval()
    loss_sum, pixels = 0.0, 0
    for inputs, ids, masks in loader:
        images, ids, labels = prepare(inputs, ids, masks, device, conditioned)
        loss_sum += criterion(model(images, ids), labels).item() * labels.numel()
        pixels += labels.numel()
    return loss_sum / pixels


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode() + b"\0" + str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def train_one(mode, train, validation, args, device, positive_weight):
    conditioned = mode == "conditioned"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = InstructionEditor().to(device)
    initial_hash = state_digest(model.state_dict())
    initial_embedding = model.instruction_embedding.weight.detach().cpu().clone()
    train_loader = make_loader(train, args.batch_size, args.seed, shuffle=True)
    val_loader = make_loader(validation, args.batch_size, args.seed)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    initial_loss = validation_loss(model, val_loader, criterion, device, conditioned)
    best_loss, best_epoch, best_state = float("inf"), None, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, pixels = 0.0, 0
        for inputs, ids, masks in train_loader:
            images, ids, labels = prepare(inputs, ids, masks, device, conditioned)
            loss = criterion(model(images, ids), labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Training loss became non-finite.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * labels.numel()
            pixels += labels.numel()
        val_loss = validation_loss(model, val_loader, criterion, device, conditioned)
        if not math.isfinite(val_loss):
            raise RuntimeError("Validation loss became non-finite.")
        history.append({"epoch": epoch, "training_weighted_bce": total / pixels,
                        "validation_weighted_bce": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"{mode} epoch {epoch:02d}: train BCE={total / pixels:.6f}; validation BCE={val_loss:.6f}", flush=True)
    selected = InstructionEditor().cpu().eval()
    selected.load_state_dict(best_state, strict=True)
    return selected, {
        "initial_state_sha256": initial_hash, "selected_state_sha256": state_digest(best_state),
        "initial_validation_loss": initial_loss, "selected_validation_loss": best_loss,
        "selected_epoch": best_epoch, "history": history,
        "initial_embedding": initial_embedding.tolist(),
        "selected_embedding": best_state["instruction_embedding.weight"].tolist(),
        "embedding_row_l2_change": (best_state["instruction_embedding.weight"] - initial_embedding).norm(dim=1).tolist(),
        "conditioning": "Requested IDs during training and inference." if conditioned
        else "Both IDs replaced with 0 during training and inference; same architecture and initialization.",
    }


@torch.no_grad()
def predict_logits(model, data, batch_size, conditioned=True):
    model.eval()
    outputs = []
    for inputs, ids, masks in make_loader(data, batch_size, seed=0):
        images, ids, _ = prepare(inputs, ids, masks, torch.device("cpu"), conditioned)
        outputs.append(model(images, ids)[:, 0].numpy())
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite prediction logits.")
    return result


def scene_indices(records):
    groups = defaultdict(dict)
    for index, record in enumerate(records):
        task = record["instruction_id"]
        if task in groups[record["scene_id"]]:
            raise ValueError("Duplicate instruction within a scene.")
        groups[record["scene_id"]][task] = index
    if not groups or any(set(pair) != {0, 1} for pair in groups.values()):
        raise ValueError("Expected paired circle/square requests.")
    return groups


def counterfactual_logits(logits, records, swap=False):
    """Identical scene inputs let us reuse the other ID's already computed output."""
    indices = scene_indices(records)
    selected = [indices[r["scene_id"]][1 - r["instruction_id"] if swap else 0] for r in records]
    return logits[selected]


def evaluate_logits(data_dir, data, logits, threshold, method):
    inputs, _, labels, records, _ = data
    predictions = apply_threshold(logits, threshold)
    totals, by_task, cases = {}, {0: {}, 1: {}}, []
    files_digest = hashlib.sha256()
    for index, (record, prediction_mask) in enumerate(zip(records, predictions, strict=True)):
        original = Image.fromarray(inputs[index].numpy())
        prediction = render_edit(original, prediction_mask)
        loaded = []
        for kind in ("target", "preserve_mask"):
            path = data_dir / record[kind]
            files_digest.update(record[kind].encode() + b"\0" + path.read_bytes())
            with Image.open(path) as image:
                loaded.append(image.copy())
        mask = Image.fromarray(labels[index].numpy() * 255)
        counts = instruction_counts(original, prediction, loaded[0], mask, loaded[1])
        for destination in (totals, by_task[record["instruction_id"]]):
            for key, value in counts.items():
                destination[key] = destination.get(key, 0) + value
        cases.append({
            "id": record["id"], "scene_id": record["scene_id"], "instruction_id": record["instruction_id"],
            "method": method, "exact": bool(counts["exact_images"]),
            "wrong_pixels": counts["pixels"] - counts["correct_pixels"],
            "missed_requested_pixels": counts["edited_pixels"] - counts["correct_edited_pixels"],
            "damaged_opposite_pixels": counts["preserved_object_pixels"] - counts["unchanged_object_pixels"],
        })
    return {
        **summarize_requests(totals), **paired_scene_scores(cases),
        "per_instruction": {name: summarize_requests(by_task[task])
                            for task, name in ((0, "circle"), (1, "square"))},
        "target_preservation_files_sha256": files_digest.hexdigest(),
    }, cases


def save_preview(data_dir, validation, logits, threshold, output):
    inputs, _, _, records, _ = validation
    cell, row_height = 128, 152
    selected = records[:4]
    sheet = Image.new("RGB", (4 * cell, 24 + len(selected) * row_height), "#e8e8e8")
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(("Input", "Conditioned", "Constant ID", "Target")):
        draw.text((column * cell + 4, 6), title, fill="black")
    for index, record in enumerate(selected):
        original = Image.fromarray(inputs[index].numpy())
        with Image.open(data_dir / record["target"]) as image:
            target = image.copy()
        panels = (original, render_edit(original, apply_threshold(logits["conditioned"][index], threshold)),
                  render_edit(original, apply_threshold(logits["instruction_blind"][index], threshold)), target)
        for column, panel in enumerate(panels):
            sheet.paste(panel.resize((cell, cell), Image.Resampling.NEAREST),
                        (column * cell, 24 + index * row_height))
        draw.text((4, 24 + index * row_height + cell + 4),
                  f"{record['scene_id']} | edit {record['requested_object']}", fill="black")
    sheet.save(output)


def main(argv=None, *, run_context=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/instruction_edits_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/reproductions/instruction_cnn_v1")
    args = parser.parse_args(argv)
    if (args.epochs <= 0 or args.batch_size <= 0 or not math.isfinite(args.learning_rate)
            or args.learning_rate <= 0 or not 0 <= args.seed < 2**32 or not 0 < args.mask_threshold < 1):
        parser.error("Epochs, batch size, learning rate, seed, or mask threshold is invalid.")
    if any((args.output_dir / name).exists() for name in ("validation.json", "conditioned.pt", "instruction_blind.pt")):
        parser.error("Output already contains an experiment; choose a fresh output directory.")
    metadata = run_metadata(args, seed=args.seed, configuration=run_context)
    manifest = (args.data_dir / "manifest.jsonl").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != args.manifest_sha256:
        parser.error("Dataset manifest differs from the declared experiment configuration.")
    records = [json.loads(line) for line in manifest.decode().splitlines()]
    train = load_instruction_split(args.data_dir, records, "train")
    validation = load_instruction_split(args.data_dir, records, "validation")
    name = args.device
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; use --device cpu.")
    device = torch.device(name)
    torch.set_num_threads(4)
    positives = int(train[2].sum().item())
    positive_weight = (train[2].numel() - positives) / positives
    models, training, logits = {}, {}, {}
    started = time.perf_counter()
    print(f"Training device: {device}; train requests: {len(train[3])}; validation requests: {len(validation[3])}", flush=True)
    for mode in ("instruction_blind", "conditioned"):
        models[mode], training[mode] = train_one(mode, train, validation, args, device, positive_weight)
        logits[mode] = predict_logits(models[mode], validation, args.batch_size, conditioned=mode == "conditioned")
    if training["conditioned"]["initial_state_sha256"] != training["instruction_blind"]["initial_state_sha256"]:
        raise ValueError("Comparison models did not start from identical weights.")
    logits["conditioned_clamped_to_circle"] = counterfactual_logits(logits["conditioned"], validation[3])
    logits["conditioned_swapped_ids"] = counterfactual_logits(logits["conditioned"], validation[3], swap=True)
    methods, cases = {}, []
    for method, values in logits.items():
        methods[method], method_cases = evaluate_logits(args.data_dir, validation, values, args.mask_threshold, method)
        cases.extend(method_cases)
    predictions = apply_threshold(logits["conditioned"], args.mask_threshold)
    pairs = scene_indices(validation[3])
    different = sum(not np.array_equal(predictions[pair[0]], predictions[pair[1]]) for pair in pairs.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for mode, model in models.items():
        path = args.output_dir / f"{mode}.pt"
        torch.save({"model_state": model.state_dict(), "model": "InstructionEditor", "seed": args.seed,
                    "conditioning_mode": mode, "selected_epoch": training[mode]["selected_epoch"],
                    "manifest_sha256": args.manifest_sha256}, path)
        checkpoints[mode] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        **metadata, "experiment": "instruction_cnn_v1", "dataset": "instruction_edits_v1", "split": "validation",
        "model": "InstructionEditor: Embedding(2,4), RGB+4 broadcast embedding channels, dilated CNN 16 channels, 1-channel mask logits",
        "parameters": sum(p.numel() for p in models["conditioned"].parameters()), "receptive_field": 31,
        "training_device": str(device), "evaluation_device": "cpu", "torch_version": str(torch.__version__),
        "epochs": args.epochs, "batch_size": args.batch_size, "optimizer": "Adam", "learning_rate": args.learning_rate,
        "loss": "BCEWithLogitsLoss; positive weight = negative/positive training mask pixels", "positive_weight": positive_weight,
        "mask_probability_threshold": args.mask_threshold, "threshold_selection_performed": False,
        "selection": "Each model independently selects lowest weighted validation BCE over the declared epochs.",
        "primary_evaluation": "Paired-scene success: both requested edits must match their full targets exactly.",
        "train_requests": len(train[3]), "validation_requests": len(validation[3]),
        "train_scenes": len(scene_indices(train[3])), "validation_scenes": len(pairs),
        "manifest_sha256": args.manifest_sha256, "training_input_mask_sha256": train[4], "validation_input_mask_sha256": validation[4],
        "checkpoint_sha256": checkpoints, "training": training, "methods": methods,
        "conditioning_diagnostics": {"scenes": len(pairs), "scenes_with_different_outputs_for_two_ids": different},
        "preview_request_ids": [r["id"] for r in validation[3][:4]],
        "test_images_loaded": False, "test_model_performance_measured": False,
        "scope": "Two categorical commands on synthetic shapes; RGB+ID inputs, mask supervision, fixed red renderer. No free-form language claim.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "validation_cases.jsonl").write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases))
    save_preview(args.data_dir, validation, logits, args.mask_threshold, args.output_dir / "preview.png")
    for method, scores in methods.items():
        print(f"{method}: exact requests={scores['counts']['exact_images']}/{scores['counts']['images']}; both edits exact={scores['scene_counts']['exact_scenes']}/{scores['scene_counts']['scenes']}", flush=True)
    print(f"Report: {args.output_dir / 'validation.json'}", flush=True)
