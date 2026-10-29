# Configured experiments and provenance

Each existing experiment has a JSON file under `configs/` and uses the same
command. For example, from the repository directory:

```bash
uv sync --locked
uv run --locked python scripts/run_experiment.py --config configs/shape_cnn_v1.json --dry-run
uv run --locked python scripts/run_experiment.py --config configs/shape_cnn_v1.json
```

The dry run validates the whole configuration and prints resolved settings.
It creates no output directory, imports no training pipeline, and needs no
dataset or checkpoint. Remove `--dry-run` to execute the declared experiment.
The runner rejects duplicate keys, unknown or missing parameters, invalid types,
non-finite numbers, invalid seeds, and nonpositive training settings.

## Configs and prerequisites

| Config | Pipeline | Prerequisite |
| --- | --- | --- |
| `configs/color_baselines.json` | Color-only validation baselines | `data/circle_edits_v2` |
| `configs/pixel_mlp_v1.json` | Train PixelMLP | `data/circle_edits_v2` |
| `configs/shape_cnn_v1.json` | Train ShapeEditor | `data/shape_edits_v1` |
| `configs/instruction_cnn_v1.json` | Train conditioned CNN and matched constant-ID control | `data/instruction_edits_v1`, with the declared manifest hash |
| `configs/pixel_diagnostics.json` | Saved PixelMLP probes | `results/pixel_mlp/model.pt` |
| `configs/shape_cnn_v1_errors.json` | Frozen CNN validation error analysis | Original spatial checkpoint, validation report, and dataset |
| `configs/shape_cnn_v1_thresholds.json` | Declared validation threshold grid | Original spatial checkpoint, validation report, and dataset |
| `configs/shape_cnn_v1_frozen_test.json` | Reproduce the frozen test evaluation | Original spatial checkpoint, dataset, and tracked selected configuration |

Generate missing datasets with the existing commands:

```bash
uv run --locked python scripts/generate_pairs.py
uv run --locked python scripts/generate_spatial_pairs.py
```

Generated datasets and trained checkpoints are excluded from Git. Training configs
write new checkpoints under `results/reproductions/`, while the analysis configs
read the original local checkpoints under `results/pixel_mlp` or
`results/spatial_cnn`. They are independent reproductions of the recorded stages,
not an automatic chain. To analyze a new training run, copy the relevant config,
change its `parameters.run_dir`, and choose a new `parameters.output_dir`.

The frozen evaluator additionally verifies the checkpoint and manifest hashes
in `docs/experiments/shape_cnn_v1_selected_config.json`. A newly trained checkpoint
cannot silently replace that frozen model. Reproducing the observed test result
is a compatibility check, not a new unseen test. Future model development needs
a newly reserved holdout. This configuration milestone does not rerun the test.

## Paths and output protection

The `--config` argument is resolved from the caller's working directory. Paths
inside the JSON resolve from the repository root, regardless of the config file's
location; absolute paths are also accepted. All effective values are saved.

The supplied output paths are separate from the original run directories. The
runner reserves a new directory and refuses to reuse any existing directory,
including an empty one. To repeat a run, copy the config and choose a fresh output
path. A failed run may leave its reserved directory behind. The older standalone
commands keep their original overwrite behavior.

## What each new report records

- `git_commit` and `git_dirty`: revision and working-tree status at the start.
  A clean checkout ties the run to that commit. With local changes, the commit
  alone is insufficient; source/config hashes describe the actual file snapshot.
  Without Git metadata, both values are `null` rather than invented.
- `effective_parameters`: parsed pipeline options, including resolved paths.
  The pre-existing numeric `parameters` field remains the model parameter count.
- `seed` and `seed_role`: the training seed controls Python, NumPy, PyTorch, and
  the shuffled loader. Deterministic evaluation uses `null` because it does not
  select random samples or train weights; model/data hashes identify its inputs.
- `configuration`: original parsed JSON, SHA-256 of its exact bytes, its path,
  and resolved parameters. For standalone commands this field is `null`.
- `source_files_sha256` and `environment_files_sha256`: package/command source
  plus `pyproject.toml` and `uv.lock`. Existing data and checkpoint hashes remain.

The two original training configs reproduce their original seed and hyperparameters.
The instruction config declares a separate paired-task experiment; its mask
threshold and manifest hash are explicit. See [its protocol](instruction_model.md).
Exact
numerics can still vary across devices, operating systems, and GPU kernels.
Historical reports are not amended with today's commit, which did not produce
their original results.

## Verification

The test suite validates every checked-in config, verifies dry runs and output
protection, and trains both models on tiny CPU fixtures with no test split. It
checks identical weights for equal seeds, different weights for a changed seed,
independent loader shuffle seeds, and report/checkpoint metadata. Git fixtures
verify clean/dirty revision tracking and the no-Git case.

```bash
uv run --locked python -m unittest discover -s tests -v
```
