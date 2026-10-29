# Frozen evaluation on the test split

## What was frozen before evaluation?

The first spatial CNN checkpoint and threshold **0.80** were identified by the
[selected configuration](experiments/shape_cnn_v1_selected_config.json), committed
in [65c2e0e](https://github.com/VishwasSP01/visual-learning-lab/commit/65c2e0ed31f6239e06959ea05113f0fd20a212e2)
before this test evaluation. The checkpoint and dataset manifest are bound to
that configuration by SHA-256 hashes.

We evaluated only this selected operating point. There was no retraining,
threshold search, or model selection using the test results. The fixed
color-replacement rule is included as the previously defined reference method.

Dataset-integrity checks and generation previews had inspected test files
earlier. Those checks did not measure model performance or select the checkpoint
or threshold; model selection and threshold comparison used validation data.

## Results on 1,000 test images

| Measurement | Color rule | Frozen CNN plus renderer |
| --- | ---: | ---: |
| Entire images correct exactly | 0 / 1,000 | **711 / 1,000 (71.1%)** |
| Circle pixels edited correctly | 100% | 99.685% |
| Square pixels preserved | 0% | 99.691% |
| Entire square preserved exactly | 0% | 91.6% |
| White background pixels preserved | 100% | 99.991% |
| Edit-mask intersection over union | 43.613% | 99.116% |

Pixel rates pool counts across all test images. Exact image success requires
every output pixel to match its target. There are 1,604 wrong pixels in total:
568 missed circle pixels, 722 unintended square edits, and 314 unintended
white-background edits. A total of 289 images contain at least one mistake.

The same operating point scored 338 / 500 (67.6%) on validation. Validation
and test contain different scenes and different held-out circle-center tiles,
so their measured percentages need not be identical.

## Example outputs

The preview shows the first four test examples in manifest order, without
selecting them by outcome. It is an illustration, not a substitute for the
aggregate scores, and does not imply that every test image is correct.

![First four test inputs, baseline, frozen CNN, and targets](assets/spatial_cnn_test_preview.png)

The [recorded test report](experiments/shape_cnn_v1_test.json) includes raw counts,
configuration/checkpoint/data fingerprints, the threshold, and the evaluator's
source fingerprint. Per-image records for both methods are saved locally.

## Reproduce the evaluation

With the original selected checkpoint and generated dataset present:

```bash
uv run --locked python -m unittest discover -s tests -v
uv run --locked python scripts/evaluate_frozen_model.py
open results/spatial_cnn/frozen_test/preview.png
```

The default configuration is the recorded, versioned selection file. Before
loading test images, the evaluator checks that the checkpoint and manifest
match its hashes, that selection was on validation, and that the threshold is
valid. It checks again that the model and configuration did not change during
evaluation. Outputs are stored under `results/spatial_cnn/frozen_test/`.

Reproducing these same frozen results is distinct from changing settings to
improve the observed test score. A retrained checkpoint may differ numerically;
it must not be presented as the exact checkpoint identified in this report.

## What this establishes

This is a complete introductory experiment: data generation, a simple baseline,
supervised learning, validation diagnostics, threshold selection, and a frozen
test evaluation. It demonstrates shape-selective editing within this particular
synthetic rendering process. It does not establish general photo editing,
language understanding, or reliability on unrelated images.

The current test results are now observed. We retain this experiment as a
recorded baseline. A subsequent instruction-conditioned experiment should use
a newly reserved holdout rather than tuning against this result.
