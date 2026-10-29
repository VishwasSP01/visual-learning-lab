"""Probe the saved model's limits without retraining or using the test set."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from visual_lab.baselines import recolor_blue
from visual_lab.metrics import score_prediction, summarize
from visual_lab.models import PixelMLP
from visual_lab.paths import ROOT
from visual_lab.provenance import run_metadata
from visual_lab.rendering import to_image


PROBES = [
    ("Blue", (0, 0, 255), True),
    ("Green", (0, 128, 0), True),
    ("White", (255, 255, 255), True),
    ("Red", (255, 0, 0), False),
    ("Black", (0, 0, 0), False),
    ("Yellow", (255, 255, 0), False),
    ("Cyan", (0, 255, 255), False),
    ("Gray", (128, 128, 128), False),
]


def save_color_preview(cases, output):
    font = ImageFont.load_default(size=14)
    image = Image.new("RGB", (650, 68 + len(cases) * 64), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 8), "Saved model: familiar colors and new colors", fill="black", font=font)
    for x, title in ((12, "Color"), (185, "Input"), (340, "Model output"), (495, "Expected")):
        draw.text((x, 40), title, fill="black", font=font)
    for row, case in enumerate(cases):
        y = 68 + row * 64
        draw.rectangle((0, y, 649, y + 63), fill="#f0f0f0" if row % 2 == 0 else "white")
        draw.text((12, y + 5), case["name"], fill="black", font=font)
        draw.text((12, y + 28), "in training" if case["seen_in_training"] else "new color",
                  fill="#555555", font=font)
        for x, field in ((185, "input"), (340, "prediction"), (495, "expected")):
            draw.rectangle((x, y + 3, x + 122, y + 29), fill=tuple(case[field]), outline="#888888")
            draw.text((x, y + 36), str(tuple(case[field])), fill="black")
    image.save(output / "colors.png")


def make_shape_probe():
    original = Image.new("RGB", (64, 64), "white")
    drawing = ImageDraw.Draw(original)
    drawing.ellipse((8, 18, 28, 38), fill="blue")
    drawing.rectangle((42, 22, 56, 36), fill="blue")
    target = original.copy()
    ImageDraw.Draw(target).ellipse((8, 18, 28, 38), fill="red")
    mask = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(mask).ellipse((8, 18, 28, 38), fill=255)
    return original, target, mask


def save_shape_preview(original, prediction, rule, target, output):
    cell = 128
    image = Image.new("RGB", (4 * cell, 184), "#e8e8e8")
    draw = ImageDraw.Draw(image)
    for column, (title, result) in enumerate(zip(
        ("Input", "Pixel model", "Color rule", "Target"),
        (original, prediction, rule, target), strict=True,
    )):
        x = column * cell
        draw.text((x + 5, 6), title, fill="black")
        image.paste(result.resize((cell, cell), Image.Resampling.NEAREST), (x, 24))
    draw.text((6, 161), "New task: recolor the circle; preserve the blue square.", fill="black")
    image.save(output / "shapes.png")


def main(argv=None, *, run_context=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/pixel_mlp")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir or args.run_dir / "diagnostics"
    metadata = run_metadata(args, configuration=run_context)
    output = args.output_dir
    checkpoint_path = args.run_dir / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = PixelMLP()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    cases = []
    with torch.no_grad():
        for name, color, seen in PROBES:
            raw = model(torch.tensor(color, dtype=torch.float32) / 255.0)
            prediction = (raw.clamp(0, 1) * 255).round().to(torch.uint8).tolist()
            # A declared diagnostic extension of the fixed-color task:
            # exact blue becomes red, while every other color should stay as-is.
            expected = (255, 0, 0) if color == (0, 0, 255) else color
            cases.append({
                "name": name, "input": list(color), "prediction": prediction,
                "expected": list(expected), "seen_in_training": seen,
                "exact_match": prediction == list(expected),
            })
        original, target, mask = make_shape_probe()
        pixels = torch.tensor(list(original.get_flattened_data()), dtype=torch.float32)
        pixels = pixels.reshape(64, 64, 3) / 255.0
        prediction = to_image(model(pixels))

    rule = recolor_blue(original)
    shape_scores = {
        "pixel_model": summarize(score_prediction(original, prediction, target, mask)),
        "color_rule": summarize(score_prediction(original, rule, target, mask)),
    }
    report = {
        **metadata,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "device": "cpu", "training_performed": False, "test_dataset_used": False,
        "color_contract": "Exact blue becomes red; all other colors remain unchanged.",
        "colors": cases,
        "shape_probe": {
            "instruction": "Change the blue circle to red. Preserve the blue square.",
            "scores": shape_scores,
            "scope": "One constructed diagnostic scene, not a population performance estimate.",
        },
        "scope": "Exploratory diagnostics beyond the original training distribution; not a new benchmark.",
    }
    output.mkdir(parents=True, exist_ok=True)
    save_color_preview(cases, output)
    save_shape_preview(original, prediction, rule, target, output)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    for seen, label in ((True, "Training colors"), (False, "New colors")):
        subset = [case for case in cases if case["seen_in_training"] == seen]
        correct = sum(case["exact_match"] for case in subset)
        print(f"{label}: {correct}/{len(subset)} exact under the stated diagnostic contract.")
    for name, scores in shape_scores.items():
        print(f"Blue-circle/blue-square probe, {name}: edit accuracy={scores['edit_pixel_accuracy']:.2%}; background preservation={scores['background_preservation']:.2%}")
    print(f"Report and previews: {output}")


if __name__ == "__main__":
    main()
