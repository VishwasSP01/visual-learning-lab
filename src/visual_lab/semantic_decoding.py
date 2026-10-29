"""Decode two exclusive object masks without materializing a full argmax map."""

import torch


def semantic_object_masks(logits):
    """Return Bx2xHxW circle/square masks, exactly matching finite 3-class argmax.

    Equal scores choose the lowest class index, matching torch.argmax:
    background beats both objects on ties, and circle beats square on ties.
    The caller must check that logits are finite before using this fast path.
    """
    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError("Expected Bx3xHxW semantic logits.")
    background, circle, square = logits.unbind(dim=1)
    return torch.stack(((circle > background) & (circle >= square),
                        (square > background) & (square > circle)), dim=1)
