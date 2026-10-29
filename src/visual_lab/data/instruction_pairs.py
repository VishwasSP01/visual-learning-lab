"""Create two different requested edits for each scene, with scene-level splits."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from visual_lab.data.circle_pairs import SIZE, SPLIT_TILES, verify_saved_pair
from visual_lab.data.spatial_pairs import make_pair as draw_scene
from visual_lab.paths import ROOT
from visual_lab.provenance import run_metadata


SEED = 20261029
SCENES_PER_SPLIT = {"train": 2500, "validation": 250, "test": 500}
TASKS = (
    {"id": 0, "object": "circle", "preserve": "square",
     "instruction": "Change the blue circle to red. Keep the blue square and background unchanged."},
    {"id": 1, "object": "square", "preserve": "circle",
     "instruction": "Change the blue square to red. Keep the blue circle and background unchanged."},
)


def make_requests(rng, split):
    images, geometry = draw_scene(rng, split)
    original, circle_target, circle_mask, square_mask = images
    square_target = original.copy()
    ImageDraw.Draw(square_target).rectangle(geometry["square_box"], fill="red")
    return (original, {"circle": circle_target, "square": square_target},
            {"circle": circle_mask, "square": square_mask}, geometry)


def audit_records(records):
    """Intentional input reuse is allowed only between the two requests for one scene."""
    groups = defaultdict(list)
    for record in records:
        if record["split"] not in SCENES_PER_SPLIT:
            raise ValueError("Unknown split.")
        groups[record["scene_id"]].append(record)
    if not groups:
        raise ValueError("No scenes found.")
    scene_counts = dict.fromkeys(SCENES_PER_SPLIT, 0)
    input_owners = {}
    for scene_id, requests in groups.items():
        if (len(requests) != 2 or any(type(r["instruction_id"]) is not int for r in requests)
                or {r["instruction_id"] for r in requests} != {0, 1}):
            raise ValueError("Each scene must have exactly one request for each instruction.")
        if len({r["split"] for r in requests}) != 1:
            raise ValueError("Requests for one scene cannot cross splits.")
        if len({(r["input"], r["input_sha256"]) for r in requests}) != 1:
            raise ValueError("Both requests must share the exact input.")
        if any(requests[0][key] != requests[1][key] for key in (
                "circle_mask", "square_mask", "circle_center", "circle_tile", "circle_box", "square_box")):
            raise ValueError("Both requests must share the same scene labels and geometry.")
        for request in requests:
            task = TASKS[request["instruction_id"]]
            if (request["instruction"] != task["instruction"]
                    or request["requested_object"] != task["object"]
                    or request["mask"] != request[task["object"] + "_mask"]
                    or request["preserve_mask"] != request[task["preserve"] + "_mask"]):
                raise ValueError("Instruction and mask assignment disagree.")
        digest = requests[0]["input_sha256"]
        if digest in input_owners:
            raise ValueError("An input image is reused by different scenes or splits.")
        input_owners[digest] = scene_id
        scene_counts[requests[0]["split"]] += 1
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Request IDs must be unique.")
    return scene_counts


def verify_request(record, output):
    """Check serialized pixels for the requested edit and the opposite-shape mask."""
    verify_saved_pair(output / record["input"], output / record["target"], output / record["mask"])
    with (Image.open(output / record["input"]) as original,
          Image.open(output / record["mask"]) as edit,
          Image.open(output / record["preserve_mask"]) as preserve):
        if preserve.mode != "L" or preserve.size != (SIZE, SIZE) or set(preserve.get_flattened_data()) != {0, 255}:
            raise ValueError("Invalid preservation mask.")
        if hashlib.sha256(original.tobytes()).hexdigest() != record["input_sha256"]:
            raise ValueError("Saved input pixels differ from the manifest hash.")
        for pixel, selected, unchanged in zip(original.get_flattened_data(),
                                               edit.get_flattened_data(),
                                               preserve.get_flattened_data(), strict=True):
            if selected and unchanged:
                raise ValueError("Edit and preservation masks overlap.")
            if (selected or unchanged) and pixel != (0, 0, 255):
                raise ValueError("Both shapes must be blue in the input.")
            if not selected and not unchanged and pixel != (255, 255, 255):
                raise ValueError("Pixels outside both objects must be white.")


def save_preview(examples, output):
    cell, row_height = 112, 140
    sheet = Image.new("RGB", (cell * 4, 24 + len(examples) * row_height), "#e8e8e8")
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(("Same input", "Requested target", "Edit mask", "Preserve mask")):
        draw.text((column * cell + 3, 6), title, fill="black")
    for row, (label, images) in enumerate(examples):
        for column, image in enumerate(images):
            sheet.paste(image.convert("RGB").resize((cell, cell), Image.Resampling.NEAREST),
                        (column * cell, 24 + row * row_height))
        draw.text((4, 24 + row * row_height + cell + 4), label, fill="black")
    sheet.save(output / "preview.png")


def generate_dataset(output, scene_counts=None, seed=SEED):
    counts = dict(SCENES_PER_SPLIT if scene_counts is None else scene_counts)
    if set(counts) != set(SCENES_PER_SPLIT) or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError("Specify a positive scene count for train, validation, and test.")
    if type(seed) is not int or not 0 <= seed < 2**32 - 2:
        raise ValueError("Seed and its two split offsets must fit in [0, 2**32).")
    output = Path(output).resolve()
    metadata = run_metadata(argparse.Namespace(output=output, scenes_per_split=counts, seed=seed), seed=seed)
    output.mkdir(parents=True, exist_ok=False)
    seen, records, preview = set(), [], []
    for split_index, split in enumerate(SCENES_PER_SPLIT):
        rng = random.Random(seed + split_index)
        for kind in ("input", "circle_mask", "square_mask", "target_circle", "target_square"):
            (output / split / kind).mkdir(parents=True)
        for index in range(counts[split]):
            for _ in range(1000):
                original, targets, masks, geometry = make_requests(rng, split)
                digest = hashlib.sha256(original.tobytes()).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    break
            else:
                raise RuntimeError("Could not generate a unique scene.")
            scene_id = f"{split}-scene-{index:04d}"
            paths = {kind: f"{split}/{kind}/{index:04d}.png" for kind in (
                "input", "circle_mask", "square_mask", "target_circle", "target_square",
            )}
            original.save(output / paths["input"])
            for name in ("circle", "square"):
                masks[name].save(output / paths[name + "_mask"])
                targets[name].save(output / paths["target_" + name])
            for task in TASKS:
                name, preserve = task["object"], task["preserve"]
                record = {
                    "id": scene_id + "-" + name, "scene_id": scene_id, "split": split,
                    "instruction_id": task["id"], "instruction": task["instruction"],
                    "requested_object": name, "input": paths["input"], "input_sha256": digest,
                    "target": paths["target_" + name], "mask": paths[name + "_mask"],
                    "preserve_mask": paths[preserve + "_mask"],
                    "circle_mask": paths["circle_mask"], "square_mask": paths["square_mask"],
                    **geometry,
                }
                verify_request(record, output)
                records.append(record)
                if index == 0 and split != "test":
                    preview.append((f"{split}: edit {name}", (original, targets[name], masks[name], masks[preserve])))
        print(f"Verified {counts[split]:,} {split} scenes / {2 * counts[split]:,} requests.", flush=True)
    if audit_records(records) != counts:
        raise ValueError("Serialized request counts disagree with the planned scene counts.")
    manifest = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records).encode()
    (output / "manifest.jsonl").write_bytes(manifest)
    metadata.update({
        "dataset": "instruction_edits_v1", "schema_version": 1, "size": SIZE,
        "seed_role": "Dataset generation; independent RNG stream per split (seed + split index).",
        "scenes_per_split": counts, "requests_per_split": {split: 2 * n for split, n in counts.items()},
        "total_scenes": len(seen), "total_requests": len(records), "tasks": TASKS,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "tiles": {split: [list(tile) for tile in sorted(tiles)] for split, tiles in SPLIT_TILES.items()},
        "split_policy": "Assign scenes to spatial splits first; both instructions remain with their scene.",
        "input_reuse": "Exactly two requests share one scene input; no input pixel hash belongs to different scenes.",
        "model_inputs": "RGB image and categorical instruction ID (0=circle, 1=square).",
        "label_policy": "Targets, masks, and geometry are supervision/audit labels, never model inputs.",
        "preview_splits": ["train", "validation"], "training_performed": False,
        "test_model_performance_measured": False,
        "scope": "Two fixed commands on the same synthetic renderer; no free-form language or real-photo claim.",
    })
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    save_preview(preview, output)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/instruction_edits_v1")
    args = parser.parse_args(argv)
    try:
        metadata = generate_dataset(args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Created {metadata['total_scenes']:,} scenes and {metadata['total_requests']:,} paired requests.")
    print(f"Training/validation preview: {args.output / 'preview.png'}")
