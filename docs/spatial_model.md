# Our first convolutional image editor

## What changed from the pixel network?

The pixel network saw one RGB value at a time. This model receives a complete
image and uses convolutional layers to examine patterns of neighboring pixels.
The same filters are shared across the image.

Our small CNN has 7,425 learned parameters. Four 3 x 3 convolutional layers use
dilations 1, 2, 4, and 8, followed by a 1 x 1 output layer. ReLU follows each
of the first four layers. The theoretical receptive field is 31 x 31 pixels:
that is the maximum input neighborhood that can influence one output pixel.
It is not a guarantee that every pixel in that neighborhood is used equally.

## What is learned, and what is fixed?

The network learns **where to edit**. It predicts one edit-region logit per
pixel, from the original RGB image alone. A fixed sigmoid threshold of 0.5
converts those predictions into a binary mask.

A separate renderer sets predicted edit pixels to red and leaves the other
pixels as they were. This red-color operation is ordinary code, not learned
image generation. An incorrect mask can therefore change square or background
pixels; the renderer does not consult ground-truth masks to fix mistakes.

Ground-truth circle masks supervise the network during training. Targets,
circle masks, and square masks are used for evaluation; none is a model input.
The instruction remains fixed, so this does not demonstrate language understanding.

## Training

The first experiment used:

- 5,000 training images and 500 validation images from `shape_edits_v1`.
- Twelve epochs, batch size 64, Adam, and learning rate 0.001.
- Binary cross-entropy with logits. Circle pixels are weighted by the ratio
  of negative to positive **training** pixels (about 21.68).
- Seed 42, PyTorch 2.15.0, and the Mac's MPS GPU backend.
- Checkpoint selection by lowest weighted validation loss; epoch 12 was selected.

Weighting addresses the imbalance between the small circle region and the
many pixels outside it. Validation loss decreased from 1.3480 to 0.00885.
The held-out 1,000-image test split was not loaded for training or evaluation.

## Measured validation results

| Measurement | Color rule | CNN plus renderer |
| --- | ---: | ---: |
| Circle pixels edited correctly | 100% | 99.86% |
| Square pixels preserved | 0% | 99.10% |
| White background pixels preserved | 100% | 99.98% |
| Entire square preserved exactly | 0% | 74.20% |
| Entire image correct exactly | 0 / 500 | 210 / 500 (42%) |
| Edit-mask intersection over union | 43.61% | 98.27% |

Pixel rates pool counts across all validation images. Exact image success
requires every output pixel to match the target. High per-pixel scores can
coexist with a lower exact-image score because a few mistakes invalidate an
otherwise good image. The saved preview uses the first four validation examples,
not a selection of the best outputs, and visibly includes boundary mistakes.

![Input, color rule, CNN edit, target, and predicted mask](assets/spatial_cnn_preview.png)

The [recorded run report](experiments/shape_cnn_v1.json) includes the configuration,
all epoch losses, raw metric counts, and data/source fingerprints. The saved
checkpoint was reloaded on CPU and independently reproduced every reported
validation count across the 500 images.

These are checkpoint-selection validation results from one seed on synthetic
data, not final test results or evidence of arbitrary shape recognition.

## Reproduce

```bash
uv sync --locked
uv run --locked python scripts/generate_spatial_pairs.py
uv run --locked python -m unittest discover -s tests -v
uv run --locked python scripts/train_spatial_model.py
open results/spatial_cnn/preview.png
```

The weights, report, and preview are generated under `results/spatial_cnn/`.
Rerunning overwrites those generated artifacts. The documented run report and
preview above are retained separately in Git. GPU numerics can vary between runs.

The subsequent [error analysis](spatial_error_analysis.md) inspects the remaining
mistakes without changing the model or training schedule. The test set stays
reserved for a later frozen experiment.

References: [PyTorch convolution](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html)
and [binary cross-entropy with logits](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html).
