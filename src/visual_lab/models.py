"""Neural architectures; state-dict keys match the original checkpoints."""

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
