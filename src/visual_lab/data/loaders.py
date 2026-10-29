"""Load supervised splits without passing labels to model inputs."""

import hashlib

import numpy as np
import torch
from PIL import Image


def load_pixel_split(data_dir, records, split):
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


def load_spatial_split(data_dir, records, split):
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
