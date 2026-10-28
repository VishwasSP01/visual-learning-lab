"""Train a small CNN to locate the editable circle, then render that region red."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_baseline import recolor_blue, score_prediction, summarize


ROOT = Path(__file__).resolve().parent
SEED = 42


class ShapeEditor(nn.Module):
    """Predict edit-region logits from RGB alone, with a 31 x 31 receptive field."""

    def __init__(self):
        super().__init__()
        layers = []
        channels = 3
        for dilation in (1, 2, 4, 8):
            layers.extend((nn.Conv2d(channels, 16, 3, padding=dilation, dilation=dilation), nn.ReLU()))
            channels = 16
        layers.append(nn.Conv2d(16, 1, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        return self.layers(images)


def load_split(data_dir, records, split):
    """Masks are supervision targets, never concatenated with the RGB input."""
    selected = [record for record in records if record["split"] == split]
    if not selected:
        raise ValueError(f"No {split} examples found.")
    inputs, masks = [], []
    digest = hashlib.sha256()
    for record in selected:
        for kind, destination in (("input", inputs), ("mask", masks)):
            path = data_dir / record[kind]
            digest.update(record[kind].encode() + b"\0" + path.read_bytes())
            with Image.open(path) as image:
                expected_mode = "RGB" if kind == "input" else "L"
                if image.mode != expected_mode or image.size != (64, 64):
                    raise ValueError(f"Invalid {kind}: {path}")
                array = np.array(image, dtype=np.uint8)
                if kind == "mask":
                    if set(np.unique(array)) != {0, 255}:
                        raise ValueError(f"Expected a non-empty binary mask: {path}")
                    array = (array == 255).astype(np.uint8)
                destination.append(array)
    return (torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(masks)),
            selected, digest.hexdigest())


def make_loader(inputs, masks, batch_size, shuffle=False):
    return DataLoader(
        TensorDataset(inputs, masks), batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(SEED), num_workers=0,
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


def render_edit(original, predicted_mask):
    """Apply a fixed red color only where the model predicts an edit."""
    pixels = np.array(original, dtype=np.uint8)
    if predicted_mask.shape != pixels.shape[:2] or predicted_mask.dtype != np.bool_:
        raise ValueError("Expected a boolean prediction mask matching the input image.")
    pixels[predicted_mask] = (255, 0, 0)
    return Image.fromarray(pixels)


def spatial_counts(original, prediction, target, mask, square_mask):
    """Measure the square separately so the white background cannot hide damage."""
    counts = score_prediction(original, prediction, target, mask)
    before, after = np.array(original), np.array(prediction)
    circle = np.array(mask) == 255
    square = np.array(square_mask) == 255
    if square_mask.mode != "L" or square.shape != circle.shape or not square.any() or (circle & square).any():
        raise ValueError("Invalid square-preservation mask.")
    same = np.all(before == after, axis=2)
    changed = np.any(before != after, axis=2)
    white_background = ~(circle | square)
    counts.update({
        "square_pixels": int(square.sum()),
        "unchanged_square_pixels": int((same & square).sum()),
        "exact_squares": int(same[square].all()),
        "white_background_pixels": int(white_background.sum()),
        "unchanged_white_background_pixels": int((same & white_background).sum()),
        "mask_intersection": int((changed & circle).sum()),
        "mask_union": int((changed | circle).sum()),
    })
    return counts


def summarize_spatial(counts):
    return {
        **summarize(counts),
        "square_preservation": counts["unchanged_square_pixels"] / counts["square_pixels"],
        "exact_square_preservation": counts["exact_squares"] / counts["images"],
        "white_background_preservation": counts["unchanged_white_background_pixels"] / counts["white_background_pixels"],
        "edit_mask_iou": counts["mask_intersection"] / counts["mask_union"],
    }


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/shape_edits_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/spatial_cnn")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, and learning rate must be positive.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; use --device cpu.")
    name = args.device
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    started = time.perf_counter()
    manifest = (args.data_dir / "manifest.jsonl").read_bytes()
    records = [json.loads(line) for line in manifest.decode().splitlines()]
    train_inputs, train_masks, train_records, train_hash = load_split(args.data_dir, records, "train")
    val_inputs, val_masks, val_records, val_hash = load_split(args.data_dir, records, "validation")
    train_loader = make_loader(train_inputs, train_masks, args.batch_size, shuffle=True)
    val_loader = make_loader(val_inputs, val_masks, args.batch_size)
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
        "experiment": "shape_cnn_v1", "dataset": "shape_edits_v1", "split": "validation",
        "architecture": "Conv2d 3->16,16->16,16->16,16->16 (3x3; dilations 1,2,4,8; ReLU), 16->1 (1x1)",
        "parameters": parameters, "receptive_field": 31, "seed": SEED,
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
    torch.save({"model_state": best_state, "architecture": report["architecture"],
                "selected_epoch": best_epoch}, args.output_dir / "model.pt")
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_preview(examples, args.output_dir / "preview.png")
    for method, scores in report["methods"].items():
        print(f"{method}: circle edit={scores['edit_pixel_accuracy']:.2%}; square preserved={scores['square_preservation']:.2%}; exact images={scores['exact_image_success']:.2%}; edit IoU={scores['edit_mask_iou']:.2%}", flush=True)
    print(f"Selected epoch {best_epoch}; report: {args.output_dir / 'validation.json'}")


if __name__ == "__main__":
    main()
