# Local image-editing demo

The demo runs the saved spatial CNN on CPU and displays one validation example
at a time. It uses Python's standard-library HTTP server with the existing
PyTorch, NumPy, and Pillow dependencies. There is no additional demo dependency.

## Launch the recorded model

From the repository directory:

```bash
uv sync --locked
uv run --locked python scripts/demo.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). The server binds to the local
loopback address. Press **Ctrl+C** in its terminal to stop it. Use `--port 7861`
if the default port is already in use.

These local files must exist:

- `data/shape_edits_v1/manifest.jsonl` and its validation images and masks.
- `results/spatial_cnn/model.pt`, the original saved checkpoint.
- `docs/experiments/shape_cnn_v1_selected_config.json`, tracked in Git.

The checkpoint and dataset manifest must match the selected configuration's
SHA-256 hashes. We load the exact checkpoint bytes that pass that check. The
original configuration uses threshold **0.80**. Missing or changed files produce
a startup error; the demo does not train a replacement or silently use other
weights. Generated data and model weights are not distributed in the repository.

## If you are starting from a fresh clone

You can create an independent local model and select its threshold on validation
data. This is a new reproduction, so its weights and threshold may differ from
the recorded checkpoint. The following commands assume fresh output directories:

```bash
uv run --locked python scripts/generate_spatial_pairs.py
uv run --locked python scripts/run_experiment.py --config configs/shape_cnn_v1.json
uv run --locked python scripts/compare_thresholds.py --run-dir results/reproductions/shape_cnn_v1 --output-dir results/reproductions/demo_thresholds
uv run --locked python scripts/demo.py --checkpoint results/reproductions/shape_cnn_v1/model.pt --config results/reproductions/demo_thresholds/selected_config.json
```

Threshold selection uses validation performance. This route never evaluates the
test split. The demo verifies the new checkpoint against its own selected config;
it does not relabel a different checkpoint as the published frozen model.

## What the page shows

Use the dropdown or Previous/Next buttons to browse validation examples in
manifest order. The first example opens automatically. All validation examples
are available, including failures; training and test examples are excluded.

The three panels show the original image, the rendered model prediction, and
the target. The network receives RGB pixels only. It predicts an edit mask, and
the fixed renderer paints selected pixels red. Targets and ground-truth masks
are read after prediction for the reference panel and scoring.

Scores describe the displayed case:

- **Whole-image result:** exact match only if every pixel equals the target.
- **Circle edited correctly:** target matches within the circle mask.
- **Square preserved:** square pixels unchanged from the original input.
- **Incorrect pixels:** all pixels differing from the target, including the
  circle, square, and white background.

These are inspection results on known validation data, not a new benchmark.
This model handles one synthetic fixed task; it does not accept text instructions
or arbitrary photo uploads. The page uses a three-column desktop layout and a
stacked layout in narrow browser panels.

The demo reads local artifacts and returns images in memory. It writes no
datasets, checkpoints, evaluation reports, or configuration files.

## Verification

Tests exercise a CPU checkpoint whose all-red prediction has hand-calculated
errors. They check image output, scores, validation-only selection, rejected
checkpoint changes, target isolation, and the local HTTP responses. They use
temporary fixtures and need no original dataset or saved checkpoint.

```bash
uv run --locked python -m unittest discover -s tests -v
```
