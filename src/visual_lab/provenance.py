"""Fingerprint the implementation as well as the thin command wrappers."""

import hashlib
import subprocess

from visual_lab.paths import ROOT


def source_hashes():
    """Record every source file now that each experiment spans several modules."""
    paths = sorted((ROOT / "src/visual_lab").rglob("*.py"))
    paths += sorted((ROOT / "scripts").glob("*.py"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def git_state(root=ROOT):
    """A commit alone identifies the run's code only when the checkout is clean."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root, capture_output=True, text=True, check=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status)}


def run_metadata(parameters, *, seed=None, configuration=None):
    """Snapshot settings and implementation before an experiment starts."""
    return {
        **git_state(),
        "seed": seed,
        "seed_role": "training RNG and shuffled data loader" if seed is not None
        else "not used: deterministic inference or evaluation",
        "effective_parameters": {
            key: str(value.resolve()) if hasattr(value, "__fspath__") else value
            for key, value in vars(parameters).items()
        },
        "configuration": configuration,
        "source_files_sha256": source_hashes(),
        "environment_files_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("pyproject.toml", "uv.lock")
        },
    }
