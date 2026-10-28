# Visual Learning Lab

A laptop-first project exploring neural networks, diffusion models, and reliable
image editing through reproducible experiments.

The goal is to learn how models change a requested part of an image while
preserving everything else. We build this in small steps on an Apple-silicon Mac.

## First milestone: build the data

Our first task is deliberately simple:

> Change the blue circle to red. Keep everything else unchanged.

Each 64 x 64 pixel scene contains a blue circle and a separate green square on a
white background. Circle and square sizes and positions vary between examples.

Each example has:

- **Input:** the original scene with a blue circle.
- **Target:** the correct result, with the circle recolored red.
- **Mask:** white pixels identify the intended edit; black pixels must stay unchanged.
- **Instruction:** the text describing the edit.

The target is drawn by ordinary Python code, before training a model.
These masks are ground truth for training and evaluation; a future editor will
receive an image and instruction, not the ground-truth mask by default.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
these commands from the repository directory:

```bash
uv sync --locked
uv run --locked python generate_pairs.py
open data/circle_edits_v2/preview.png
```

The script creates **6,500 examples**:

| Split | Examples | Purpose |
| --- | ---: | --- |
| Training | 5,000 | Learn the transformation |
| Validation | 500 | Choose settings and monitor overfitting |
| Test | 1,000 | Evaluate the selected model |

These counts are an initial experiment budget, not a claim that more data is
always better. Learning curves will guide later changes.

The manifest at `data/circle_edits_v2/manifest.jsonl` lists the files, geometry,
and instruction for every example. Paths in the manifest are relative to
`data/circle_edits_v2/`. `metadata.json` records the seed and split protocol.

The base random seed is fixed at 42, with an independent stream for each split.
Input images are deduplicated across all three splits. The script checks the
saved files to confirm that every selected pixel changes from blue to red and
every other pixel stays identical.

### Held-out positions

Circle centers occupy a 4 x 4 grid of spatial tiles. Validation reserves the
top-right and bottom-left tiles; test reserves the top-left and bottom-right
tiles. Training uses the other twelve tiles. Each split therefore has distinct
circle-center positions. Circle sizes and green-square layouts vary as well.

Nearby positions can still be visually similar. This is a modest check of
spatial generalization within one synthetic drawing process, not a claim that
all similar scenes or other forms of leakage have been eliminated. A future
model's data loader must respect these manifest splits rather than reshuffle
the whole dataset.

Regenerating the dataset overwrites its generated files. The generated `data/`
directory and local Python environment are excluded from Git; the code and
dependency lockfile are enough to rebuild this milestone. The earlier 100-pair
debugging dataset remains in `data/circle_edits/`; version 2 is stored separately.

## How to read the preview

Each row shows one example. There are two rows per split, labeled underneath.
The columns are the input image, target image, and edit mask. The green square
must be identical before and after the edit.

## Baseline: check the evaluation before learning

A baseline is a simple reference solution. We evaluate two methods:

- **Unchanged:** return the input image without editing anything.
- **Color rule:** replace every exact blue RGB pixel `(0, 0, 255)` with red
  `(255, 0, 0)`. This method receives only the input image.

Neither method uses the target image, ground-truth mask, geometry, or instruction
to produce its output. The evaluator reads targets and masks to score results.

Run the baseline and its hand-calculated metric checks:

```bash
uv run --locked python -m unittest -v test_evaluate_baseline
uv run --locked python evaluate_baseline.py
open results/baselines/validation_preview.png
```

The default evaluation uses the **500 validation examples**. The test set is
reserved for later. The report at `results/baselines/validation.json` contains
raw counts, scores, and hashes identifying the evaluated data. Generated results
are excluded from Git and can be reproduced with the command above.

We measure:

- **Exact image success:** fraction of images where every pixel matches the target.
- **Edit pixel accuracy:** fraction of pixels inside the mask matching the target.
- **Background preservation:** fraction of pixels outside the mask left unchanged.
- **Whole-image pixel accuracy:** fraction of all pixels matching the target.

Pixel rates pool counts across images. This matters because most pixels are
background: an unchanged image can score highly on whole-image accuracy while
completely failing the requested edit. The checks use three-pixel examples with
hand-calculated answers, including a deliberately broken all-red output.

Measured on the current 500-example validation split:

| Method | Exact image success | Edit pixel accuracy | Background preservation | Whole-image pixel accuracy |
| --- | ---: | ---: | ---: | ---: |
| Unchanged | 0% | 0% | 100% | 94.38% |
| Color rule | 100% | 100% | 100% | 100% |

The rule solves this controlled task perfectly. A neural network will be a
learning exercise, not an improvement over this rule. Later tasks will need
additional ambiguity and variation before model complexity can be justified.

## First neural network: learn a pixel-color mapping

We now have a tiny PyTorch network: **3 RGB inputs -> 16 hidden units -> 3 RGB
outputs**, with ReLU between the layers and **115 learned parameters**.
It applies the same learned transformation independently to each pixel.

```bash
uv sync --locked
uv run --locked python train_pixel_model.py
open results/pixel_mlp/preview.png
```

The default run trains on 5,000 images for ten epochs with Adam, batch size 64,
and learning rate 0.01. It uses the Mac GPU through MPS when available, otherwise
CPU. `--device cpu` explicitly selects CPU. No external service is required.

Inputs and targets are normalized from `[0, 255]` to `[0, 1]` in each batch.
Only input pixels enter the network. Targets supply the training loss; masks
are used only for scoring the selected model. No test images are loaded.

The first local run selected epoch 6 using the lowest validation MSE:

| Measurement | Observed result |
| --- | ---: |
| Initial validation MSE | 1.5150364 |
| Selected validation MSE | approximately 1.4e-17 |
| Exact validation images after 8-bit conversion | 500 / 500 |
| Edit pixel accuracy | 100% |
| Background preservation | 100% |

The experiment writes the selected weights to `results/pixel_mlp/model.pt` and
the configuration, dataset hashes, epoch losses, and scores to
`results/pixel_mlp/validation.json`. Rerunning overwrites these generated files.
Seeds and dependencies are fixed; exact GPU numerics can vary between runs.

The very small loss reflects the task's limited palette: only three distinct
input colors. Position holdouts do not challenge this pixelwise architecture,
which never sees coordinates. Neither object recognition nor instruction
understanding is demonstrated. This exercise teaches learning and optimization;
it does not outperform the perfect fixed rule on this controlled task.

Read [the short learning notes](docs/first_neural_network.md) for an explanation
of pixels, parameters, loss, backpropagation, and epochs.

## Inspecting generalization

```bash
uv run --locked python inspect_pixel_model.py
open results/pixel_mlp/diagnostics/colors.png
open results/pixel_mlp/diagnostics/shapes.png
```

These exploratory diagnostics use the saved weights without retraining or
accessing the held-out test dataset:

- All three familiar input colors are handled correctly.
- All five unfamiliar colors tried are incorrectly changed under the declared
  diagnostic contract (exact blue becomes red; all other colors stay unchanged).
- In one new scene containing a blue circle and a blue square, both the learned
  model and the color rule recolor the square incorrectly. Each changes 225
  pixels that should be preserved, while correctly editing the circle.

These are constructed probes, not population-level benchmark scores. They
show why the original perfect validation score should not be interpreted as
general editing ability. See [the explanation](docs/generalization.md) and
`results/pixel_mlp/diagnostics/report.json` for the inputs, outputs, and scope.

## Next dataset: edit by shape rather than color

```bash
uv run --locked python generate_spatial_pairs.py
open data/shape_edits_v1/preview.png
```

This separate dataset has 5,000 training, 500 validation, and 1,000 test examples.
Both shapes are now blue, and their bounding-box widths and heights match.
The target recolors only the circle, preserving the blue square and white
background. Sizes and positions vary, with the same held-out circle-position
tile protocol used above.

The manifest adds a `square_mask` so the next experiment can measure preservation
of the distractor separately from the large white background. Masks and geometry
are ground-truth labels; they must not be provided as model inputs.
The generator checks the actual saved files for correct edits, preservation,
and non-overlapping masks. The CNN experiment below learns from this dataset.

See [the spatial dataset explanation](docs/spatial_dataset.md) for the task,
checks, split protocol, and limitations. The preview shows one example per split:
input, target, circle edit mask, and square preservation mask.

## First spatial model: learn where to edit

```bash
uv run --locked python -m unittest discover -v
uv run --locked python train_spatial_model.py
open results/spatial_cnn/preview.png
```

A small **7,425-parameter CNN** receives only the RGB image and predicts an
edit-region mask. Its convolutional layers have a theoretical 31 x 31 receptive
field. Ground-truth masks are supervision labels, not input channels. A separate
fixed renderer paints the predicted region red; the color operation is not learned.

The first run trained for twelve epochs on 5,000 examples and selected epoch 12
by lowest weighted validation loss. The 1,000 test examples were not loaded.

| Validation measurement | Color rule | CNN plus renderer |
| --- | ---: | ---: |
| Circle pixels edited correctly | 100% | 99.86% |
| Square pixels preserved | 0% | 99.10% |
| Entire images correct | 0 / 500 | 210 / 500 (42%) |
| Edit-mask IoU | 43.61% | 98.27% |

The distinction between per-pixel accuracy and exact images matters: a few
incorrect pixels still make the whole image incorrect. The first four validation
examples below include visible boundary errors; they were not cherry-picked.

![Input, color rule, CNN edit, target, predicted mask](docs/assets/spatial_cnn_preview.png)

See [the CNN explanation](docs/spatial_model.md) and
[recorded experiment](docs/experiments/shape_cnn_v1.json) for training settings,
raw counts, checkpoint verification, and limitations. The recorded preview and
small JSON report are versioned; generated datasets and model weights remain
excluded from Git.

## Scope and next step

This is synthetic learning data, not evidence of real-photo editing capability.
The splits share the same drawing process and instruction; they do not establish
generalization to new object types, instructions, or real images. In particular,
the original color-only task can be solved by a simple color-replacement rule,
which remains a baseline when interpreting neural-network results. Learning a
single fixed transformation does not demonstrate language understanding.

Next we will inspect the convolutional model's remaining mistakes before changing
its architecture or training schedule. Larger generative models are later
experiments, not a requirement for finishing the first milestones.

The original `draw_circle.py`, `blue_circle.png`, and `red_circle.png` are kept
as the first hand-created examples.
