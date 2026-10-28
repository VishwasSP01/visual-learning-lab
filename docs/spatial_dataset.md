# Learning to use surrounding pixels

## The new task

Every input has a blue circle and a blue square on a white background.
Only the circle should become red. The square must remain blue.

The objects have matching bounding-box widths and heights in each scene,
and their positions and sizes vary. Circle radii range from 5 through 9 pixels,
so a matching square can fit beside a circle in any allowed center position.
Their bounding boxes do not overlap.
This removes color and bounding-box size as shortcuts for identifying which
object to edit. Shape information is present in the surrounding pixels.

## The data

`data/shape_edits_v1/` contains 5,000 training, 500 validation, and 1,000 test
examples, each at 64 x 64 pixels. The earlier color-only datasets remain separate.

Each example includes:

- `input`: the original image with two blue shapes.
- `target`: the desired red-circle/blue-square image.
- `mask`: the circle region that should change.
- `square_mask`: the blue square region that must remain unchanged.
- `instruction`: the same fixed request for every scene.
- Geometry and hashes for auditing the generated examples.

Masks and geometry are labels for supervision and evaluation, not model inputs.
A model will receive the original image. This is still a fixed-task experiment;
there is no evidence of understanding variable instructions.

## Checks before training

The generator opens the saved files and verifies that:

- Every circle pixel changes from exact blue to exact red.
- Every square pixel stays exact blue.
- Every other pixel stays white.
- Edit and square-preservation masks are non-empty and do not overlap.
- Input images are distinct across the three splits.

The same spatial split protocol is used as before: circle centers occupy a
4 x 4 tile grid. Validation reserves two corner tiles, test reserves the other
two, and training uses the remaining twelve. Only circle positions are held
out this way; square positions, rendering rules, and sizes are shared.

These are modest spatial holdouts. They do not establish generalization to
different rendering styles, rotations, overlapping objects, or photographs.

## Why the next model needs context

An independent pixel-color mapping receives the same input `(0, 0, 255)` in
both objects. It must output the same color for both, so it cannot solve this
new task perfectly. Adding more original color-only data cannot change that
architectural limitation.

The next experiment will investigate a small convolutional network with access
to neighboring pixels. Its effective receptive field must be large enough to
provide useful shape context. Evaluation will measure both successful circle
editing and square preservation; overall image accuracy alone is insufficient.

Rebuild the dataset:

```bash
uv run --locked python generate_spatial_pairs.py
```

Regeneration overwrites this dataset's generated files. No model has been
trained on this new task at this milestone.
