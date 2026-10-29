# Local categorical image-editing demo

The default demo uses the recorded semantic editor on CPU. Pick one validation
scene, then choose **Edit circle** or **Edit square** while preserving the other
object and the white background. All 250 validation scenes and both choices are
available, including failures. The page displays input, prediction, target, and
scores for the displayed request. It uses the existing dependencies and Python's
standard-library HTTP server.

## Launch the recorded model

From the repository directory:

```bash
uv sync --locked
uv run --locked python scripts/demo.py
```

Open [the local demo](http://127.0.0.1:7860). The server binds only to loopback.
Press **Ctrl+C** in its terminal to stop it. Use `--port 7861` if needed.

Required local files:

- `data/instruction_edits_v1/manifest.jsonl` and its validation images and masks.
- `results/reproductions/semantic_dilated_v2/model.pt`, the saved seed-42 weights.
- `docs/experiments/semantic_dilated_v2_selected_config.json`, tracked in Git.

Startup verifies checkpoint and manifest hashes, model architecture, class
mapping, and decoder. It loads the exact weight bytes that passed verification.
Missing or changed artifacts fail visibly; the server does not train replacement
weights. Generated data and model weights are excluded from Git.

The checkpoint is the one used in the [frozen test](instruction_test_evaluation.md).
The demo serves validation scenes only; its per-example scores are not a new
benchmark. The selected config's historical false test-performance flag records
that selection happened before the frozen test.

## README animation

The [README GIF](assets/instruction_demo.gif) shows `validation-scene-0000`
first with the circle choice, then the square choice. Its frames are captured
from the running local browser demo, with the input and model-result panels
cropped and arranged side by side. Each state lasts 2.5 seconds, looping every
5 seconds. The captured input pixels are identical between choices. This is
one illustrative validation scene, not an additional benchmark.

## What the model receives

The network receives only RGB pixels. It predicts background/circle/square
classes, using the frozen argmax decoder. Your categorical choice selects the
predicted object region, and a fixed renderer paints it red. Targets and masks
are opened after prediction, supplying the comparison panel and scores only.
This does not interpret free-form language or accept arbitrary photo uploads.

Scores adapt to the request:

- **Whole-image result:** every pixel must match for an exact result.
- **Selected object edited correctly:** requested-region pixels matching target.
- **Other object preserved:** opposite-object pixels unchanged from input.
- **Incorrect pixels:** all mismatches, including white-background damage.

Changing the instruction keeps the same input scene. Previous/Next and the scene
selector preserve the selected instruction. The desktop layout has three image
columns; narrow panels stack them. The server writes no data, models, reports,
or configs, and performs no training. API requests accept only known validation
scene IDs and the two available choices.

## If starting from a fresh clone

Reproductions create new local weights; exact GPU numerics can differ. Generate
the instruction data, reproduce the reference, and choose its cutoff using
validation, then point a copied semantic config at that local reference config:

```bash
uv run --locked python scripts/generate_instruction_pairs.py
uv run --locked python scripts/run_experiment.py --config configs/instruction_cnn_v2.json
uv run --locked python scripts/run_experiment.py --config configs/instruction_cnn_v2_thresholds.json
uv run --locked python - <<'CONFIG'
import json
from pathlib import Path
config = json.loads(Path("configs/semantic_dilated_v2.json").read_text())
config["experiment"] = "local_semantic_demo"
config["parameters"]["reference_config"] = "results/reproductions/instruction_cnn_v2_thresholds/selected_config.json"
config["parameters"]["output_dir"] = "results/reproductions/local_semantic_demo"
Path("results/local_semantic_demo.json").write_text(json.dumps(config, indent=2) + "\n")
CONFIG
uv run --locked python scripts/run_experiment.py --config results/local_semantic_demo.json
uv run --locked python scripts/demo.py --checkpoint results/reproductions/local_semantic_demo/model.pt --config results/reproductions/local_semantic_demo/selected_config.json
```

These commands assume new output directories. This route does not evaluate test
performance or relabel reproduced weights as the published frozen checkpoint.
It selects the reference cutoff and semantic checkpoint from validation only.

## Original circle-only demo

With the original spatial dataset/checkpoint available:

```bash
uv run --locked python scripts/demo.py --mode spatial
```

This mode uses `data/shape_edits_v1`, `results/spatial_cnn/model.pt`, and
`docs/experiments/shape_cnn_v1_selected_config.json` by default, with the original
0.80 cutoff and 500 validation examples. Only the circle choice is available.
Custom spatial checkpoints/configs require the same `--mode spatial` flag.
See the [spatial experiment](spatial_model.md) and [config guide](experiment_configs.md)
for independent reproductions of that earlier model.

## Verification

Temporary-fixture tests verify both choices with hand-calculated errors, same
RGB inputs, label isolation, rejected artifact changes, validation-only access,
CPU inference, and HTTP responses. The semantic fixture's train/test image
folders are removed entirely. Legacy spatial-mode tests remain in place.

The browser check switches circle to square, confirms the updated target and
preservation labels, and checks Next while keeping the square choice. An audit
of all 500 actual validation requests through the demo reproduces the saved
491 exact edits, 244 exact pairs, and 20 wrong pixels. No test model pass is
performed by these demo checks.

```bash
uv run --locked python -m unittest discover -s tests -v
```
