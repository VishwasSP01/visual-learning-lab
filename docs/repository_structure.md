# Repository structure and experiment compatibility

The code now lives in an editable Python package under `src/visual_lab/`.
The commands under `scripts/` call that package; tests import the same package.
There are no `sys.path` patches or required `PYTHONPATH` settings.

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked python scripts/train_spatial_model.py --help
```

The command filenames and experiment defaults are retained. Add `scripts/` to
the old command paths. A shared repository path keeps the default datasets,
checkpoints, configurations, and reports in their original locations.
This layout is intended for work in a repository checkout installed in editable
mode; it does not bundle datasets or checkpoints into a distributable wheel.

The refactor preserves the model layers and state-dict keys, data generation,
metric definitions, rendering, and threshold decisions. Training is still
implemented in separate pixel and spatial experiment pipelines. The recorded
checkpoint and validation-selected 0.80 threshold remain the same experiment.

The reports and preview images in `docs/experiments/` and `docs/assets/` are
historical records. Their source hashes identify the code at the commits that
produced them; moving code does not retroactively change those hashes. Retrieve
the original root-level scripts from commit `89092d4` when auditing that source.
Newly generated reports also include `source_files_sha256`, covering package
modules and command wrappers, because a single script no longer contains the
entire implementation.

Verification compares the pre-refactor and packaged code using the existing
checkpoints on validation data, including logits, rendered edits, and raw metric
counts. This is a compatibility check, with no training or threshold search.
The frozen test benchmark remains recorded at **711 / 1,000 exact images**.

The compatibility audit found:

- Identical loader tensors and model outputs for all 500 validation images in
  each of the pixel and spatial experiments.
- Identical rendered spatial edits and raw metric counts, reproducing the
  recorded **338 / 500** exact validation images at threshold 0.80.
- Identical pixels and geometry for 96 deterministic generated scenes per
  generator, and 42 pre-existing function/class definitions preserved in the
  package (apart from renaming the two split loaders).
- Successful help output from all nine experiment commands when invoked from
  outside the repository. This check is also part of the unit-test suite.
- All 15 tests passing in a separate source copy with a fresh environment and
  no generated datasets or saved model weights.
- Unchanged historical JSON reports, preview images, saved checkpoints, and
  dataset manifests, verified by SHA-256 hashes.
