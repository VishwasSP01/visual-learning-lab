# What did our first model actually learn?

The first model reproduced all 500 validation images exactly after conversion
to 8-bit RGB. Those images still contained only three input colors: exact blue,
green, and white. We now inspect the same saved weights on a few deliberately
different inputs. No retraining or held-out test dataset is involved.

## Diagnostic 1: unfamiliar colors

For this diagnostic, we declare a broader desired behavior: exact blue should
become red, and every other color should be preserved. This expectation extends
the original dataset, which provided no examples of other colors.

The saved model handles all three training colors correctly. It changes each
of the five unfamiliar colors we tried: red, black, yellow, cyan, and gray.
For example, a red pixel `(255, 0, 0)` becomes `(133, 127, 98)`.

These eight chosen color probes are a small diagnostic, not an estimate of
performance over all possible colors. The full inputs and outputs are recorded
in `results/pixel_mlp/diagnostics/report.json`.

## Diagnostic 2: two blue shapes

We create a blue circle beside a blue square, with a red-circle/blue-square
target. The requested edit is to recolor only the circle.

Both the neural network and the fixed color rule incorrectly turn the square
red too. The circle is edited correctly, but 225 pixels of the square are also
changed. Exact image success is therefore zero on this constructed scene.

Every pure-blue pixel receives exactly the same prediction from our network.
It has no neighboring pixels or shape information, so it cannot give different
answers to blue pixels based on which shape contains them. More training
images from the original task cannot supply this missing information.

## The lesson

A model can score perfectly within a narrow data distribution and still fail
when the task changes. We need both suitable training examples and a model
that can access the information needed to solve the task.

The next experiment will require spatial context: scenes where blue appears in
both an object that should change and an object that must remain unchanged.
A convolutional model is a reasonable next architecture to investigate, with
a receptive field large enough to access useful surrounding pixels. Success
must still be measured; changing the architecture alone does not guarantee it.

Reproduce these diagnostics:

```bash
uv run --locked python scripts/inspect_pixel_model.py
```

The diagnostic contract is fixed in the script. All predictions use the saved
weights, with no edits to those weights or to the original experiment results.
