# First instruction-conditioned CNN

The model receives an RGB image and one categorical request: ID 0 edits the
circle; ID 1 edits the square. A learned `Embedding(2, 4)` maps each ID to four
numbers. Those numbers become constant spatial channels beside RGB, giving the
CNN seven input channels. Backpropagation updates the embedding and convolutions
together. This is categorical conditioning, not free-form language understanding.

The CNN retains four 16-channel 3 × 3 convolutional layers with dilations 1, 2,
4, and 8, ReLU activations, and a 1 × 1 mask-logit head. Its receptive field is
31 × 31; the architecture has **8,009 parameters**, including eight embedding
values. A fixed renderer paints the predicted region red.

Targets, geometry, edit masks, and preservation masks are labels. They are never
concatenated with the image or supplied to the forward method. Label masks
supply the loss; targets and preservation masks supply evaluation only.

## Protocol declared before the full run

| Setting | Value |
| --- | --- |
| Dataset | `instruction_edits_v1`, bound by manifest SHA-256 in the config |
| Training | 2,500 scenes / 5,000 requests |
| Validation | 250 scenes / 500 requests |
| Seed | 42 |
| Epochs | 12 for each of two models |
| Batch size | 64 |
| Optimizer | Adam, learning rate 0.001 |
| Loss | Weighted BCE with positive weight from training masks only |
| Checkpoint selection | Each model's lowest weighted validation BCE |
| Mask threshold | 0.50, fixed before evaluation; no sweep in this experiment |
| Evaluation device | CPU after training, using the selected weights |

The model with instructions and its control use the same architecture, seed,
starting weights, training examples, batch order, optimization, and epoch budget.
The control replaces every instruction with ID 0 during both training and
inference. It therefore learns from the same conflicting paired labels without
receiving the information needed to distinguish them. Starting state hashes are
required to match; the selected states and embedding changes are also recorded.

The differentiable mask loss selects the checkpoint. The primary reported
behavior is **paired-scene success**: both instructions must match their complete
targets exactly. Exact-request success, requested-region accuracy, and
opposite-object/background preservation are reported separately, with counts for
circle and square requests. A high pixel rate cannot establish a successful edit.

## Comparisons

Four methods are evaluated on the same validation requests:

- **Conditioned:** the trained model receives the actual requested IDs.
- **Instruction-blind:** the separately trained, matched model always receives
  ID 0. Any fixed output for one image can satisfy at most one of its two targets.
- **Conditioned, clamped to circle:** the trained conditioned model receives ID 0
  for both requests. This checks whether removing the choice changes behavior.
- **Conditioned, swapped IDs:** the trained conditioned model receives the other
  request's ID while scoring still uses the originally requested target.

Each scene has byte-identical inputs under both requests. The two counterfactual
checks reuse that scene's already computed output for the selected ID; no target
or mask is used to choose the output. A test verifies equivalence with direct
forward passes under changed IDs. The report also counts scenes where changing
the ID changes the rendered output. Changed outputs alone do not prove correct
edits, so paired-scene success remains the primary behavioral check.

## Run and inspect

The dataset must already exist. Generate it once with
`uv run --locked python scripts/generate_instruction_pairs.py` if needed. Then:

```bash
uv run --locked python scripts/run_experiment.py --config configs/instruction_cnn_v1.json
open results/reproductions/instruction_cnn_v1/preview.png
```

The output directory must be new. The run writes `conditioned.pt`,
`instruction_blind.pt`, `validation.json`, `validation_cases.jsonl`, and a preview
of the first two validation scenes under both requests. Checkpoints contain CPU
weights; the report includes config, revision, data/source hashes, training
history, selected epochs, embedding changes, and all four comparisons.

The training loader permits only training and validation image access. Tests
delete every test image before an end-to-end fixture run, demonstrating that
the pipeline does not depend on those files. The new 500-scene test set stays
reserved. This experiment does not replace the original fixed-circle benchmark.

## First 12-epoch result and next declared run

The first run used clean revision
[`eaf3573`](https://github.com/VishwasSP01/visual-learning-lab/commit/eaf3573a0bc95c07f9206fc40cdadb3222bd8a1d).
The conditioned model gets **12 / 500 requests exact (2.4%)**, and **0 / 250
scenes** have both requests exact. The constant-ID control gets zero exact
requests and zero exact scene pairs. These results are retained in the
[original validation report](experiments/instruction_cnn_v1_validation.json).

Changing the instruction changes the conditioned output on all 250 validation
scenes, and swapping the IDs reduces exact requests to zero. This confirms an
effect of the ID, but does not establish reliable editing. Opposite-object
preservation is only 81.41% overall, with 63.92% for square-edit requests.

Validation loss is still improving: 0.12467 at epoch 8 and 0.04830 at epoch 12.
Before measuring any test performance, a separate `instruction_cnn_v2` config
extends **both models to 24 epochs**. It keeps the architecture, seed, optimizer,
batch size, loss, manifest, and 0.50 threshold unchanged. This is a validation-led
training-budget change; the first result is not overwritten or presented as a
successful editor. The config and this rationale are committed before that run.

```bash
uv run --locked python scripts/run_experiment.py --config configs/instruction_cnn_v2.json
```

The first two validation scenes under the original 12-epoch model:

![First validation scene pairs after twelve epochs](assets/instruction_cnn_v1_preview.png)
