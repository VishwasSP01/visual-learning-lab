"""Train a small CNN to locate the editable circle, then render that region red."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from visual_lab.baselines import recolor_blue
from visual_lab.data.loaders import load_spatial_split as load_split
from visual_lab.metrics import spatial_counts, summarize_spatial
from visual_lab.models import ShapeEditor
from visual_lab.paths import ROOT
from visual_lab.provenance import run_metadata
from visual_lab.rendering import render_edit


SEED = 42


def make_loader(inputs, masks, batch_size, shuffle=False, seed=SEED):
    return DataLoader(
        TensorDataset(inputs, masks), batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )


def prepare_batch(inputs, masks, device):
    images = inputs.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).contiguous() / 255
    labels = masks.unsqueeze(1).to(device=device, dtype=torch.float32)
    return images, labels


@torch.no_grad()
def validation_loss(model, loader, criterion, device):
    model.eval()
    total, pixels = 0.0, 0
    for inputs, masks in loader:
        images, labels = prepare_batch(inputs, masks, device)
        total += criterion(model(images), labels).item() * labels.numel()
        pixels += labels.numel()
    return total / pixels


@torch.no_grad()
def predict_masks(model, inputs, device):
    model.eval()
    images = inputs.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).contiguous() / 255
    # Fixed decision threshold: sigmoid(logit) >= 0.5, equivalently logit >= 0.
    return (model(images)[:, 0] >= 0).cpu().numpy()


def save_preview(examples, output):
    cell, row_height = 112, 128
    sheet = Image.new("RGB", (cell * 5, 24 + len(examples) * row_height), "#e8e8e8")
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(("Input", "Color rule", "CNN edit", "Target", "Predicted mask")):
        draw.text((column * cell + 3, 6), title, fill="black")
    for row, images in enumerate(examples):
        for column, image in enumerate(images):
            sheet.paste(image.convert("RGB").resize((cell, cell), Image.Resampling.NEAREST),
                        (column * cell, 24 + row * row_height))
    sheet.save(output)


def main(argv=None, *, run_context=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/shape_edits_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/spatial_cnn")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must be in [0, 2**32).")
    metadata = run_metadata(args, seed=args.seed, configuration=run_context)
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, and learning rate must be positive.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; use --device cpu.")
    name = args.device
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    started = time.perf_counter()
    manifest = (args.data_dir / "manifest.jsonl").read_bytes()
    records = [json.loads(line) for line in manifest.decode().splitlines()]
    train_inputs, train_masks, train_records, train_hash = load_split(args.data_dir, records, "train")
    val_inputs, val_masks, val_records, val_hash = load_split(args.data_dir, records, "validation")
    train_loader = make_loader(train_inputs, train_masks, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_inputs, val_masks, args.batch_size, seed=args.seed)
    positives = int(train_masks.sum().item())
    negatives = train_masks.numel() - positives
    positive_weight = negatives / positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    model = ShapeEditor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    initial_loss = validation_loss(model, val_loader, criterion, device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}; parameters: {parameters:,}; train: {len(train_records):,}; validation: {len(val_records):,}", flush=True)
    print(f"Initial validation weighted BCE: {initial_loss:.6f}", flush=True)
    best_loss, best_epoch, best_state = float("inf"), None, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, pixels = 0.0, 0
        for inputs, masks in train_loader:
            images, labels = prepare_batch(inputs, masks, device)
            loss = criterion(model(images), labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Training loss became non-finite.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * labels.numel()
            pixels += labels.numel()
        val_loss = validation_loss(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_weighted_bce_during_epoch": total / pixels,
                        "validation_weighted_bce": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"Epoch {epoch:02d}: train BCE={total / pixels:.6f}; validation BCE={val_loss:.6f}", flush=True)

    model.load_state_dict(best_state)
    totals = {"color_rule": {}, "spatial_cnn": {}}
    examples = []
    for start in range(0, len(val_records), args.batch_size):
        masks = predict_masks(model, val_inputs[start:start + args.batch_size], device)
        for offset, predicted_mask in enumerate(masks):
            index = start + offset
            record = val_records[index]
            original = Image.fromarray(val_inputs[index].numpy())
            loaded = []
            for kind in ("target", "mask", "square_mask"):
                with Image.open(args.data_dir / record[kind]) as image:
                    loaded.append(image.copy())
            target, mask, square_mask = loaded
            predictions = {"color_rule": recolor_blue(original),
                           "spatial_cnn": render_edit(original, predicted_mask)}
            for method, prediction in predictions.items():
                counts = spatial_counts(original, prediction, target, mask, square_mask)
                for key, value in counts.items():
                    totals[method][key] = totals[method].get(key, 0) + value
            if len(examples) < 4:
                examples.append((original, predictions["color_rule"], predictions["spatial_cnn"],
                                 target, Image.fromarray(predicted_mask.astype(np.uint8) * 255)))

    report = {
        **metadata,
        "experiment": "shape_cnn_v1", "dataset": "shape_edits_v1", "split": "validation",
        "architecture": "Conv2d 3->16,16->16,16->16,16->16 (3x3; dilations 1,2,4,8; ReLU), 16->1 (1x1)",
        "parameters": parameters, "receptive_field": 31,
        "device": str(device), "torch_version": str(torch.__version__),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "optimizer": "Adam", "learning_rate": args.learning_rate,
        "loss": "BCEWithLogitsLoss; positive weight = negative/positive training pixels",
        "positive_weight": positive_weight, "mask_probability_threshold": 0.5,
        "initial_validation_loss": initial_loss, "selected_validation_loss": best_loss,
        "selected_epoch": best_epoch,
        "selection": "Lowest weighted validation BCE among requested epochs; test set not loaded.",
        "train_examples": len(train_records), "validation_examples": len(val_records),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "training_input_mask_sha256": train_hash, "validation_input_mask_sha256": val_hash,
        "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "renderer": "Set pixels selected by predicted mask to RGB (255,0,0); preserve all others. No ground-truth mask at inference.",
        "scope": "Fixed-task supervised localization on synthetic circles and squares; color rendering is deterministic, not learned.",
        "reproducibility": "Fixed seed and dependency lock; GPU numerics may vary.",
        "history": history, "methods": {key: summarize_spatial(value) for key, value in totals.items()},
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"seed": args.seed, "model_state": best_state, "architecture": report["architecture"],
                "selected_epoch": best_epoch}, args.output_dir / "model.pt")
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_preview(examples, args.output_dir / "preview.png")
    for method, scores in report["methods"].items():
        print(f"{method}: circle edit={scores['edit_pixel_accuracy']:.2%}; square preserved={scores['square_preservation']:.2%}; exact images={scores['exact_image_success']:.2%}; edit IoU={scores['edit_mask_iou']:.2%}", flush=True)
    print(f"Selected epoch {best_epoch}; report: {args.output_dir / 'validation.json'}")


if __name__ == "__main__":
    main()
