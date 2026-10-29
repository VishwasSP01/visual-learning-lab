"""Run a declared experiment without editing Python or assembling CLI flags."""

import argparse
import hashlib
import importlib
import json
import math
import re
from pathlib import Path

from visual_lab.paths import ROOT


TRAINING = {
    "epochs": int, "batch_size": int, "learning_rate": float,
    "device": str, "seed": int, "data_dir": Path, "output_dir": Path,
}
SPATIAL_ANALYSIS = {"data_dir": Path, "run_dir": Path, "output_dir": Path}
COMMANDS = {
    "train_pixel_model": ("pixel_training", TRAINING),
    "train_spatial_model": ("spatial_training", TRAINING),
    "train_instruction_model": ("instruction_training", {
        **TRAINING, "mask_threshold": float, "manifest_sha256": str,
    }),
    "evaluate_baseline": ("baseline_evaluation", {
        "data_dir": Path, "split": str, "output_dir": Path,
    }),
    "inspect_pixel_model": ("pixel_diagnostics", {"run_dir": Path, "output_dir": Path}),
    "analyze_spatial_errors": ("spatial_errors", SPATIAL_ANALYSIS),
    "compare_thresholds": ("threshold_comparison", SPATIAL_ANALYSIS),
    "compare_instruction_thresholds": ("instruction_thresholds", SPATIAL_ANALYSIS),
    "evaluate_frozen_model": ("frozen_evaluation", {**SPATIAL_ANALYSIS, "config": Path}),
}


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def load_configuration(path, root=ROOT):
    """Fail on unknown or incomplete settings, rather than silently using defaults."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    config = json.loads(raw, object_pairs_hook=unique_keys, parse_constant=reject_constant)
    required = {"schema_version", "experiment", "command", "parameters"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"Configuration must contain exactly: {sorted(required)}")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported schema_version; expected 1.")
    if not isinstance(config["experiment"], str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", config["experiment"]):
        raise ValueError("Experiment must be a short lowercase identifier.")
    command = config["command"]
    if not isinstance(command, str) or command not in COMMANDS:
        raise ValueError(f"Unknown command; choose one of {sorted(COMMANDS)}")
    module, schema = COMMANDS[command]
    values = config["parameters"]
    if not isinstance(values, dict) or set(values) != set(schema):
        raise ValueError(f"Parameters for {command} must contain exactly: {sorted(schema)}")
    resolved = {}
    for key, kind in schema.items():
        value = values[key]
        if kind in (Path, str):
            valid = isinstance(value, str) and bool(value.strip())
        elif kind is int:
            valid = type(value) is int
        else:
            valid = type(value) in (int, float) and math.isfinite(value)
        if not valid:
            raise ValueError(f"Invalid value for {key}: {value!r}")
        if key in ("epochs", "batch_size", "learning_rate") and value <= 0:
            raise ValueError(f"{key} must be positive.")
        if key == "seed" and not 0 <= value < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if key == "device" and value not in ("auto", "cpu", "mps"):
            raise ValueError("device must be auto, cpu, or mps.")
        if key == "split" and value not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test.")
        if key == "mask_threshold" and not 0 < value < 1:
            raise ValueError("mask_threshold must be strictly between 0 and 1.")
        if key == "manifest_sha256" and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest.")
        resolved[key] = str((root / value).resolve()) if kind is Path else value
    return module, resolved, {
        "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
        "declared": config, "resolved_parameters": resolved,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate and show settings without running or writing files.")
    args = parser.parse_args(argv)
    try:
        module, parameters, context = load_configuration(args.config)
        if args.dry_run:
            print(json.dumps(context, indent=2, sort_keys=True))
            return
        # Reserve a new output directory before importing or starting the pipeline.
        # The standalone legacy commands retain their original overwrite behavior.
        Path(parameters["output_dir"]).mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    flags = []
    for key, value in parameters.items():
        flags.extend(("--" + key.replace("_", "-"), str(value)))
    pipeline = importlib.import_module("visual_lab.experiments." + module)
    pipeline.main(flags, run_context=context)
