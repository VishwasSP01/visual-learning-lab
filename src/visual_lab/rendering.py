"""Deterministic RGB conversion and mask rendering."""

import numpy as np
import torch
from PIL import Image


def to_image(pixels):
    """Use the same declared 8-bit conversion for every model prediction."""
    array = (pixels.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    return Image.fromarray(array)


def render_edit(original, predicted_mask):
    """Apply a fixed red color only where the model predicts an edit."""
    pixels = np.array(original, dtype=np.uint8)
    if predicted_mask.shape != pixels.shape[:2] or predicted_mask.dtype != np.bool_:
        raise ValueError("Expected a boolean prediction mask matching the input image.")
    pixels[predicted_mask] = (255, 0, 0)
    return Image.fromarray(pixels)
