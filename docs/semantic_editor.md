# Joint perception for more accurate, cheaper categorical editing

## Protocol declared before the full experiments

The previous binary editor still damages circles during square requests. This
experiment changes the decomposition: learn a shared **background / circle /
square** class map from RGB, then let the categorical instruction select the
predicted circle or square class. Three-class argmax gives mutually exclusive
regions without a mask-confidence search. A fixed renderer paints the selected
region red. This is a specialized perception model plus explicit task routing,
not a learned instruction embedding or language understanding.

Training labels are built from the paired masks already available in the
training split. Images enter the model; masks never enter its forward method.
Each of the 2,500 training scenes is processed once per epoch, instead of twice
under different requests. Validation retains the same 250 scenes / 500 requests.
No new data, pretrained weights, augmentation, or test images are used.

### Three fixed comparisons

| Config | Architecture | Objective |
| --- | --- | --- |
| `semantic_dilated_v1` | Four 16-channel dilated convolutions, 3-class head; 7,459 parameters | Class-balanced cross entropy |
| `semantic_lite_ce_v1` | Small encoder/decoder with skip connections and separable residual blocks; 5,963 parameters | Same cross entropy |
| `semantic_lite_dice_v1` | Identical small model and initial weights to the previous row | Same cross entropy + 0.5 foreground soft Dice loss |

The small encoder/decoder uses widths 8, 12, 16 at full, half, and quarter
resolution. Two skip connections restore fine detail after bilinear upsampling
(`align_corners=False`). Each residual block uses two depthwise 3x3 plus
pointwise 1x1 convolutions. No normalization, dropout, or stochastic augmentation
is used. The original models/checkpoints are retained unchanged.

All three use seed **42**, **40 epochs**, batch size **64**, and Adam at **0.001**.
Each selects its own checkpoint by the lowest validation training-objective
loss, ties keeping the earliest epoch. Class weights are
`training_pixels / (3 * training_class_pixels)` and are fixed from training only.
Weighted cross entropy uses a pixel mean. Optional Dice averages the two
foreground classes for each image, with linear denominator and smoothing 1.

Rank the three selected checkpoints by exact validation scene pairs, then exact
requests, then fewer wrong pixels, then fewer convolution MACs, then config name.
Repeat the selected recipe from scratch with **seeds 7 and 2026**, keeping all
other settings fixed. Keep every result and do not choose a new recipe based on
the repeat outcomes. This tests limited seed sensitivity on the same validation
split; it does not create a new independent holdout.

The historical reference is the saved 24-epoch instruction model with the
validation-selected **0.95 cutoff**: 311/500 requests and 103/250 scene pairs.
Its checkpoint, config, manifest, and evaluated file hashes are checked. It is
a practical before/after comparison. The new three-class supervision, loss,
number of unique scene passes, and epoch budget differ, so the historical
comparison cannot isolate the effect of architecture. The two small-model
runs isolate the added Dice term under otherwise matched settings.

### Efficiency and correctness checks

Final scoring runs on CPU after loading selected weights. Report exact requests
and scene pairs, each task separately, all error-region counts, improvements and
regressions, and first-four and explicitly worst-four validation previews.

Measure parameter counts, serialized checkpoint sizes, and convolution MACs
for one request and both requests on the same scene. MACs exclude bias,
activations, interpolation, and other non-convolution operators; one MAC is not
one FLOP. Measure CPU model-plus-decoding latency with four threads, 10 warmups
and 100 samples per operation. Alternate timing order and report median/p90.
The reference batches both requests; the semantic model reuses one scene pass.
File loading, normalization, and RGB rendering are excluded from timings.
These are local measurements rather than general deployment speed guarantees.

All commands use the normal config runner:

```bash
uv run --locked python scripts/run_experiment.py --config configs/semantic_dilated_v1.json
uv run --locked python scripts/run_experiment.py --config configs/semantic_lite_ce_v1.json
uv run --locked python scripts/run_experiment.py --config configs/semantic_lite_dice_v1.json
```

The reserved instruction test set remains unopened for model performance.
Validation has already been used repeatedly for development; improvements need
a later frozen evaluation before claims about new-scene generalization.

## Design references

- [U-Net](https://arxiv.org/abs/1505.04597): contracting/expanding paths and
  fine-resolution skip connections for localization. This is a much smaller
  adapted architecture, not a reproduction of that paper.
- [MobileNets](https://arxiv.org/abs/1704.04861): depthwise separable convolutions
  as a compute-saving building block. Lower MACs do not automatically mean
  lower latency on every backend.
- [V-Net](https://arxiv.org/abs/1606.04797): overlap-based training for imbalanced
  segmentation. Our optional linear-denominator foreground Dice variant is
  specified above and is not the exact original V-Net objective.
- Implementation uses the installed PyTorch 2.15.0 reference docstrings for
  `Conv2d(groups=...)`, `cross_entropy`, `interpolate`, and `one_hot`. No new
  dependency is introduced.

## Initial three-model result and replication selection

All three seed-42 runs used clean revision
[`6ccf34e`](https://github.com/VishwasSP01/visual-learning-lab/commit/6ccf34edeabdf0ea9b96a116e04ab3037fd0abcc).
They evaluated the same validation inputs and labels; each reproduces the
reference's 311 exact requests and 103 exact scene pairs at its saved cutoff.

| Model | Selected epoch | Exact requests / 500 | Exact scene pairs / 250 | Wrong pixels |
| --- | ---: | ---: | ---: | ---: |
| Historical binary reference, cutoff 0.95 | 24 | 311 (62.2%) | 103 (41.2%) | 1,216 |
| **Joint dilated, balanced CE** | **39** | **472 (94.4%)** | **230 (92.0%)** | **104** |
| Small encoder/decoder, balanced CE | 40 | 395 (79.0%) | 154 (61.6%) | 285 |
| Same small model, CE + 0.5 Dice | 40 | 420 (84.0%) | 180 (72.0%) | 338 |

Dice improves exact outputs for the small model but increases total wrong pixels.
The small-model initial state hashes match, verifying equal initialization.
The declared ranking selects **the joint dilated model**. Its recipe is now
repeated with seeds 7 and 2026 in committed configs, without changing settings.

The depthwise encoder/decoder uses only 6.77 million convolution MACs per scene,
versus 30.28 million for the joint dilated model. Nevertheless, measured CPU
single-request medians are roughly 3.9 ms versus 0.58 ms. Operator overhead and
backend behavior matter; fewer MACs are insufficient evidence of lower latency.
The dilated model's initial timing is slightly slower for one request than the
reference (0.58 vs 0.53 ms), but faster for both requests on one scene
(0.58 vs 0.94 ms). These are local measurements with the protocol above.

Reports: [joint dilated](experiments/semantic_dilated_v1_validation.json),
[small CE](experiments/semantic_lite_ce_v1_validation.json),
[small CE + Dice](experiments/semantic_lite_dice_v1_validation.json).
