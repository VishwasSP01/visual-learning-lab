"""Check honest revision tracking in clean, modified, and non-Git source trees."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from visual_lab.provenance import git_state


class ProvenanceChecks(unittest.TestCase):
    def test_clean_and_dirty_revisions_are_distinguished(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def git(*args):
                return subprocess.run(["git", *args], cwd=root, check=True,
                                      capture_output=True, text=True).stdout.strip()
            git("init", "-q")
            git("config", "user.name", "Test Fixture")
            git("config", "user.email", "fixture@example.invalid")
            (root / "source.txt").write_text("original")
            git("add", "source.txt")
            git("-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
            revision = git("rev-parse", "HEAD")
            self.assertEqual(git_state(root), {"git_commit": revision, "git_dirty": False})
            (root / "source.txt").write_text("modified")
            self.assertEqual(git_state(root), {"git_commit": revision, "git_dirty": True})

    def test_source_without_git_records_unavailable_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(git_state(temporary), {"git_commit": None, "git_dirty": None})


if __name__ == "__main__":
    unittest.main()
