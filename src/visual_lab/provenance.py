"""Fingerprint the implementation as well as the thin command wrappers."""

import hashlib

from visual_lab.paths import ROOT


def source_hashes():
    """Record every source file now that each experiment spans several modules."""
    paths = sorted((ROOT / "src/visual_lab").rglob("*.py"))
    paths += sorted((ROOT / "scripts").glob("*.py"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}
