"""Learn a tiny RGB-to-RGB mapping, then evaluate on validation images only."""

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

from evaluate_baseline import score_prediction, summarize


ROOT = Path(__file__).resolve().parent
SEED = 42


class PixelMLP(nn.Module):
    """Apply the same small neural network independently to every RGB pixel."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, pixels):
        return self.layers(pixels)


def load_split(data_dir, records, split):
    """Load input/target pairs only; neither masks nor test images enter training."""
    selected = [record for record in records if record["split"] == split]
    if not selected:
        raise ValueError(f"No {split} examples found; generate the dataset first.")
    inputs, targets = [], []
    digest = hashlib.sha256()
    for record in selected:
        for kind, destination in (("input", inputs), ("target", targets)):
            relative = record[kind]
            path = data_dir / relative
            digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes())
            with Image.open(path) as image:
                if image.mode != "RGB" or image.size != (64, 64):
                    raise ValueError(f"Expected a 64 x 64 RGB image: {path}")
                destination.append(np.array(image, dtype=np.uint8))
    # Keep the full dataset in compact uint8 form. Normalize only each batch.
    return (
        torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(targets)),
        selected, digest.hexdigest(),
    )


def make_loader(inputs, targets, batch_size, shuffle=False):
    return DataLoader(
        TensorDataset(inputs, targets), batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(SEED), num_workers=0,
    )


@torch.no_grad()
def validation_mse(model, loader, device):
    model.eval()
    squared_error, values = 0.0, 0
    for inputs, targets in loader:
        inputs = inputs.to(device=device, dtype=torch.float32) / 255.0
        targets = targets.to(device=device, dtype=torch.float32) / 255.0
        squared_error += nn.functional.mse_loss(model(inputs), targets, reduction="sum").item()
        values += targets.numel()
    return squared_error / values


def to_image(pixels):
    """Use the same declared 8-bit conversion for every model prediction."""
    array = (pixels.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    return Image.fromarray(array)


@torch.no_grad()
def sample_predictions(model, inputs, device):
    model.eval()
    return [to_image(output) for output in model(inputs.to(device).float() / 255.0)]


def save_preview(originals, before, after, targets, output):
    cell, row_height = 112, 128
    sheet = Image.new("RGB", (4 * cell, 24 + len(before) * row_height), "#e8e8e8")
    drawing = ImageDraw.Draw(sheet)
    for column, label in enumerate(("Input", "Before learning", "After learning", "Target")):
        drawing.text((column * cell + 3, 6), label, fill="black")
    for row in range(len(before)):
        images = (Image.fromarray(originals[row].numpy()), before[row], after[row],
                  Image.fromarray(targets[row].numpy()))
        for column, image in enumerate(images):
            sheet.paste(image.resize((cell, cell), Image.Resampling.NEAREST),
                        (column * cell, 24 + row * row_height))
    sheet.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/circle_edits_v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/pixel_mlp")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, and learning rate must be positive.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; use --device cpu.")
    device_name = args.device
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    started = time.perf_counter()

    manifest = (args.data_dir / "manifest.jsonl").read_bytes()
    records = [json.loads(line) for line in manifest.decode("utf-8").splitlines()]
    train_inputs, train_targets, train_records, train_hash = load_split(args.data_dir, records, "train")
    val_inputs, val_targets, val_records, val_hash = load_split(args.data_dir, records, "validation")
    train_loader = make_loader(train_inputs, train_targets, args.batch_size, shuffle=True)
    val_loader = make_loader(val_inputs, val_targets, args.batch_size)
    model = PixelMLP().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    initial_mse = validation_mse(model, val_loader, device)
    before = sample_predictions(model, val_inputs[:3], device)
    print(f"Device: {device}; parameters: {parameters}; train: {len(train_records):,}; validation: {len(val_records):,}", flush=True)
    print(f"Before training: validation MSE={initial_mse:.8f}", flush=True)
    best_mse, best_epoch, best_state = float("inf"), None, None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        error, values = 0.0, 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device=device, dtype=torch.float32) / 255.0
            targets = targets.to(device=device, dtype=torch.float32) / 255.0
            prediction = model(inputs)
            loss = nn.functional.mse_loss(prediction, targets)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Training loss became non-finite.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            error += loss.item() * targets.numel()
            values += targets.numel()
        val_mse = validation_mse(model, val_loader, device)
        history.append({"epoch": epoch, "train_mse_during_epoch": error / values,
                        "validation_mse": val_mse})
        if val_mse < best_mse:
            best_mse, best_epoch = val_mse, epoch
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}
        print(f"Epoch {epoch:02d}: train MSE={error / values:.8f}; validation MSE={val_mse:.8f}", flush=True)

    model.load_state_dict(best_state)
    after = sample_predictions(model, val_inputs[:3], device)
    totals = {}
    # Ground-truth masks are loaded only here, to evaluate the selected model.
    with torch.no_grad():
        for start in range(0, len(val_records), args.batch_size):
            predictions = sample_predictions(model, val_inputs[start:start + args.batch_size], device)
            for offset, prediction in enumerate(predictions):
                index = start + offset
                with Image.open(args.data_dir / val_records[index]["mask"]) as mask:
                    counts = score_prediction(
                        Image.fromarray(val_inputs[index].numpy()), prediction,
                        Image.fromarray(val_targets[index].numpy()), mask,
                    )
                for key, count in counts.items():
                    totals[key] = totals.get(key, 0) + count
    scores = summarize(totals)
    report = {
        "model": "PixelMLP: Linear(3,16), ReLU, Linear(16,3)",
        "parameters": parameters, "seed": SEED, "device": str(device),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "optimizer": "Adam", "learning_rate": args.learning_rate,
        "train_examples": len(train_records), "validation_examples": len(val_records),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "training_input_target_sha256": train_hash,
        "validation_input_target_sha256": val_hash,
        "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "initial_validation_mse": initial_mse,
        "selected_epoch": best_epoch, "selected_validation_mse": best_mse,
        "selection": "Lowest validation MSE among the requested epochs; no test evaluation.",
        "image_conversion": "Clamp normalized RGB to [0,1], multiply by 255, round to uint8.",
        "scope": "Independent pixel-color mapping; no spatial reasoning or instruction understanding.",
        "reproducibility": "Seed and dependencies fixed; GPU runs may differ numerically.",
        "history": history, "validation_scores": scores,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "architecture": report["model"],
                "selected_epoch": best_epoch}, args.output_dir / "model.pt")
    report_path = args.output_dir / "validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_preview(val_inputs[:3], before, after, val_targets[:3], args.output_dir / "preview.png")
    print(f"Selected epoch {best_epoch}; validation MSE={best_mse:.8f}.", flush=True)
    print(f"Exact images={scores['exact_image_success']:.2%}; edit accuracy={scores['edit_pixel_accuracy']:.2%}; background preserved={scores['background_preservation']:.2%}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
