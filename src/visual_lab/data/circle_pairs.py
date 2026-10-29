"""Create reproducible circle-editing pairs with held-out circle positions."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from visual_lab.paths import ROOT


OUTPUT = ROOT / "data" / "circle_edits_v2"
SIZE = 64
SEED = 42
SPLITS = {"train": 5_000, "validation": 500, "test": 1_000}
INSTRUCTION = "Change the blue circle to red. Keep everything else unchanged."
# A 4 x 4 grid of 10-pixel tiles covers circle centers from 12 through 51.
# Each tile belongs to exactly one split; the two diagonals' corner tiles
# are held out. A model still sees nearby positions, so this is a modest
# spatial-generalization check, not a claim about arbitrary new scenes.
HELD_OUT_TILES = {
    "validation": {(0, 3), (3, 0)},
    "test": {(0, 0), (3, 3)},
}
TRAIN_TILES = {
    (x, y) for x in range(4) for y in range(4)
} - set.union(*HELD_OUT_TILES.values())
SPLIT_TILES = {"train": TRAIN_TILES, **HELD_OUT_TILES}


def overlaps(first, second):
    """Bounding boxes use inclusive pixel coordinates, as Pillow does."""
    left, top, right, bottom = first
    other_left, other_top, other_right, other_bottom = second
    return not (
        right < other_left or other_right < left
        or bottom < other_top or other_bottom < top
    )


def make_pair(rng, split):
    """Draw two separate objects and change only the circle's color."""
    tile_x, tile_y = rng.choice(sorted(SPLIT_TILES[split]))
    center_x = 12 + tile_x * 10 + rng.randrange(10)
    center_y = 12 + tile_y * 10 + rng.randrange(10)
    radius = rng.randint(5, 11)
    circle_box = (
        center_x - radius, center_y - radius,
        center_x + radius, center_y + radius,
    )

    for _ in range(1_000):
        side = rng.randint(8, 18)
        sx = rng.randint(2, SIZE - side - 3)
        sy = rng.randint(2, SIZE - side - 3)
        square_box = (sx, sy, sx + side, sy + side)
        if not overlaps(circle_box, square_box):
            break
    else:
        raise RuntimeError("Could not place the green square without overlap.")

    original = Image.new("RGB", (SIZE, SIZE), "white")
    drawing = ImageDraw.Draw(original)
    drawing.rectangle(square_box, fill=(0, 128, 0))
    drawing.ellipse(circle_box, fill=(0, 0, 255))

    target = original.copy()
    ImageDraw.Draw(target).ellipse(circle_box, fill=(255, 0, 0))

    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).ellipse(circle_box, fill=255)
    geometry = {
        "circle_center": [center_x, center_y],
        "circle_tile": [tile_x, tile_y],
        "circle_box": list(circle_box),
        "square_box": list(square_box),
    }
    return (original, target, mask), geometry


def verify_saved_pair(input_path, target_path, mask_path):
    """Check the actual saved training files, including the unchanged background."""
    with Image.open(input_path) as original, Image.open(target_path) as target:
        with Image.open(mask_path) as mask:
            if original.mode != "RGB" or target.mode != "RGB" or mask.mode != "L":
                raise ValueError("Unexpected image mode.")
            if any(image.size != (SIZE, SIZE) for image in (original, target, mask)):
                raise ValueError("Unexpected image dimensions.")
            if set(mask.get_flattened_data()) != {0, 255}:
                raise ValueError("The mask must have both a foreground and background.")
            for before, after, selected in zip(
                original.get_flattened_data(), target.get_flattened_data(),
                mask.get_flattened_data(), strict=True,
            ):
                if selected == 255:
                    if before != (0, 0, 255) or after != (255, 0, 0):
                        raise ValueError("An intended edit has the wrong colors.")
                elif before != after:
                    raise ValueError("A pixel outside the edit region changed.")


def save_preview(examples, output):
    """Show two examples per split: input, target, and edit mask."""
    cell = 128
    row_height = 152
    sheet = Image.new("RGB", (3 * cell, 24 + len(examples) * row_height), "#e8e8e8")
    drawing = ImageDraw.Draw(sheet)
    for column, heading in enumerate(("Input", "Target", "Edit mask")):
        drawing.text((column * cell + 6, 6), heading, fill="black")
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
    seen = set()
    preview = []
    records = []
    output.mkdir(parents=True, exist_ok=True)

    for split_index, (split, count) in enumerate(SPLITS.items()):
        rng = random.Random(SEED + split_index)
        for folder in ("input", "target", "mask"):
            (output / split / folder).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            for _ in range(1_000):
                images, geometry = make_pair(rng, split)
                digest = hashlib.sha256(images[0].tobytes()).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    break
            else:
                raise RuntimeError("Could not generate a distinct input image.")
            paths = {}
            for kind, image in zip(("input", "target", "mask"), images, strict=True):
                relative = Path(split) / kind / f"{index:04d}.png"
                image.save(output / relative)
                paths[kind] = relative.as_posix()
            verify_saved_pair(*(output / paths[kind] for kind in ("input", "target", "mask")))
            records.append({
                "id": f"{split}-{index:04d}", "split": split,
                "instruction": INSTRUCTION, "input_sha256": digest,
                **geometry, **paths,
            })
            if index < 2:
                preview.append((split, images))
        print(f"Created and verified {count:,} {split} pairs.", flush=True)

    manifest = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    (output / "manifest.jsonl").write_text(manifest, encoding="utf-8")
    metadata = {
        "schema_version": 2, "size": SIZE, "seed": SEED, "counts": SPLITS,
        "split_protocol": "disjoint_circle_center_tiles",
        "center_tile_origin": 12, "center_tile_width": 10,
        "tiles": {split: [list(tile) for tile in sorted(tiles)]
                  for split, tiles in SPLIT_TILES.items()},
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    save_preview(preview, output)
    print(f"Total: {len(records):,} distinct image pairs (base seed={SEED}).")
    print("Verified: the circle changes from blue to red; every other pixel stays identical.")
    print(f"Preview: {output / 'preview.png'}")


if __name__ == "__main__":
    main()
