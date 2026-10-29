"""Neural architectures; state-dict keys match the original checkpoints."""

import torch
from torch import nn


class PixelMLP(nn.Module):
    """Apply the same small neural network independently to every RGB pixel."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, pixels):
        return self.layers(pixels)


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


class InstructionEditor(nn.Module):
    """Broadcast a learned four-value task embedding alongside the RGB channels."""

    def __init__(self):
        super().__init__()
        self.instruction_embedding = nn.Embedding(2, 4)
        layers = []
        channels = 3 + 4
        for dilation in (1, 2, 4, 8):
            layers.extend((nn.Conv2d(channels, 16, 3, padding=dilation, dilation=dilation), nn.ReLU()))
            channels = 16
        layers.append(nn.Conv2d(16, 1, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, images, instruction_ids):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected a batch of RGB images in NCHW order.")
        if instruction_ids.ndim != 1 or instruction_ids.shape[0] != images.shape[0] or instruction_ids.dtype != torch.long:
            raise ValueError("Expected one integer instruction ID per image.")
        embedding = self.instruction_embedding(instruction_ids)
        planes = embedding[:, :, None, None].expand(-1, -1, images.shape[2], images.shape[3])
        return self.layers(torch.cat((images, planes), dim=1))
