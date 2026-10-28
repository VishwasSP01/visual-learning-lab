"""Build a shape-dependent editing task: recolor a circle, preserve a blue square."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from generate_pairs import SIZE, SEED, SPLITS, SPLIT_TILES, overlaps, verify_saved_pair


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/shape_edits_v1"
INSTRUCTION = "Change the blue circle to red. Keep the blue square and background unchanged."


def make_pair(rng, split):
    tile_x, tile_y = rng.choice(sorted(SPLIT_TILES[split]))
    center_x = 12 + tile_x * 10 + rng.randrange(10)
    center_y = 12 + tile_y * 10 + rng.randrange(10)
    # This bound leaves room for a matching square even beside a central circle.
    radius = rng.randint(5, 9)
    circle_box = (
        center_x - radius, center_y - radius,
        center_x + radius, center_y + radius,
    )
    # Match bounding-box dimensions so size alone does not identify the circle.
    side = 2 * radius
    for _ in range(1_000):
        x = rng.randint(2, SIZE - side - 3)
        y = rng.randint(2, SIZE - side - 3)
        square_box = (x, y, x + side, y + side)
        if not overlaps(circle_box, square_box):
            break
    else:
        raise RuntimeError("Could not place two non-overlapping shapes.")

    original = Image.new("RGB", (SIZE, SIZE), "white")
    drawing = ImageDraw.Draw(original)
    drawing.rectangle(square_box, fill="blue")
    drawing.ellipse(circle_box, fill="blue")
    target = original.copy()
    ImageDraw.Draw(target).ellipse(circle_box, fill="red")
    edit_mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(edit_mask).ellipse(circle_box, fill=255)
    square_mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(square_mask).rectangle(square_box, fill=255)
    geometry = {
        "circle_center": [center_x, center_y],
        "circle_tile": [tile_x, tile_y],
        "circle_box": list(circle_box), "square_box": list(square_box),
    }
    return (original, target, edit_mask, square_mask), geometry


def verify_saved_scene(record, output):
    paths = {kind: output / record[kind] for kind in ("input", "target", "mask", "square_mask")}
    verify_saved_pair(paths["input"], paths["target"], paths["mask"])
    with (
        Image.open(paths["input"]) as original,
        Image.open(paths["target"]) as target,
        Image.open(paths["mask"]) as edit_mask,
        Image.open(paths["square_mask"]) as square_mask,
    ):
        if square_mask.mode != "L" or square_mask.size != (SIZE, SIZE):
            raise ValueError("Invalid square-preservation mask.")
        if set(square_mask.get_flattened_data()) != {0, 255}:
            raise ValueError("Square mask must have foreground and background.")
        for before, after, edit, square in zip(
            original.get_flattened_data(), target.get_flattened_data(),
            edit_mask.get_flattened_data(), square_mask.get_flattened_data(), strict=True,
        ):
            if edit and square:
                raise ValueError("Edit and preservation regions overlap.")
            if square and (before != (0, 0, 255) or after != before):
                raise ValueError("The blue square must remain unchanged.")
            if not edit and not square and (before != (255, 255, 255) or after != before):
                raise ValueError("The white background must remain unchanged.")


def save_preview(examples, output):
    cell, row_height = 128, 152
    sheet = Image.new("RGB", (cell * 4, 24 + len(examples) * row_height), "#e8e8e8")
    drawing = ImageDraw.Draw(sheet)
    for column, title in enumerate(("Input", "Target", "Edit circle", "Preserve square")):
        drawing.text((column * cell + 4, 6), title, fill="black")
    for row, (split, images) in enumerate(examples):
        for column, image in enumerate(images):
            enlarged = image.convert("RGB").resize((cell, cell), Image.Resampling.NEAREST)
            sheet.paste(enlarged, (column * cell, 24 + row * row_height))
        drawing.text((6, 24 + row * row_height + cell + 4), split, fill="black")
    sheet.save(output / "preview.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seen, records, preview = set(), [], []
    kinds = ("input", "target", "mask", "square_mask")
    for split_index, (split, count) in enumerate(SPLITS.items()):
        rng = random.Random(SEED + split_index)
        for kind in kinds:
            (output / split / kind).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            for _ in range(1_000):
                images, geometry = make_pair(rng, split)
                digest = hashlib.sha256(images[0].tobytes()).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    break
            else:
                raise RuntimeError("Could not generate a unique input scene.")
            record = {
                "id": f"{split}-{index:04d}", "split": split,
                "instruction": INSTRUCTION, "input_sha256": digest, **geometry,
            }
            for kind, image in zip(kinds, images, strict=True):
                relative = Path(split) / kind / f"{index:04d}.png"
                image.save(output / relative)
                record[kind] = relative.as_posix()
            verify_saved_scene(record, output)
            records.append(record)
            if index == 0:
                preview.append((split, images))
        print(f"Created and verified {count:,} {split} shape-editing pairs.", flush=True)

    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    metadata = {
        "dataset": "shape_edits_v1", "schema_version": 1,
        "size": SIZE, "seed": SEED, "counts": SPLITS,
        "instruction": INSTRUCTION, "matched_object_bounding_box_sizes": True,
        "circle_radius_range_inclusive": [5, 9],
        "split_protocol": "disjoint_circle_center_tiles",
        "center_tile_origin": 12, "center_tile_width": 10,
        "tiles": {split: [list(tile) for tile in sorted(tiles)]
                  for split, tiles in SPLIT_TILES.items()},
        "mask_policy": "Ground truth for supervision/scoring; never model input.",
        "scope": "Synthetic, non-overlapping circles and squares; one fixed edit instruction.",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    save_preview(preview, output)
    print(f"Total: {len(records):,} unique scenes. Verified circle-only edits and square preservation.")
    print(f"Preview: {output / 'preview.png'}")


if __name__ == "__main__":
    main()
