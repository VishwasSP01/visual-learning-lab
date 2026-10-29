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


class SemanticDilatedEditor(nn.Module):
    """Classify background/circle/square jointly with the original dilated trunk."""

    def __init__(self):
        super().__init__()
        layers, channels = [], 3
        for dilation in (1, 2, 4, 8):
            layers.extend((nn.Conv2d(channels, 16, 3, padding=dilation, dilation=dilation), nn.ReLU()))
            channels = 16
        layers.append(nn.Conv2d(16, 3, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        return self.layers(images)


class SeparableResidual(nn.Module):
    """Two depthwise/pointwise convolutions with a same-width residual path."""

    def __init__(self, channels):
        super().__init__()
        layers = []
        for _ in range(2):
            layers.extend((nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                           nn.Conv2d(channels, channels, 1), nn.ReLU()))
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        return features + self.layers(features)


class SemanticLiteUNet(nn.Module):
    """Small multiscale RGB segmenter; no targets or task labels enter forward."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), SeparableResidual(8))
        self.down1 = nn.Sequential(nn.Conv2d(8, 12, 3, stride=2, padding=1), nn.ReLU(), SeparableResidual(12))
        self.down2 = nn.Sequential(nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.ReLU(), SeparableResidual(16))
        self.up1 = nn.Sequential(nn.Conv2d(16 + 12, 12, 1), nn.ReLU(), SeparableResidual(12))
        self.up2 = nn.Sequential(nn.Conv2d(12 + 8, 8, 1), nn.ReLU(), SeparableResidual(8))
        self.head = nn.Conv2d(8, 3, 1)

    def forward(self, images):
        from torch.nn.functional import interpolate

        fine = self.stem(images)
        middle = self.down1(fine)
        coarse = self.down2(middle)
        up = interpolate(coarse, size=middle.shape[-2:], mode="bilinear", align_corners=False)
        up = self.up1(torch.cat((up, middle), dim=1))
        up = interpolate(up, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.up2(torch.cat((up, fine), dim=1)))
