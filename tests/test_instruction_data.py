"""Verify paired instructions, scene isolation, and actual saved edit pixels."""

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from visual_lab.data.circle_pairs import SPLIT_TILES
from visual_lab.data.instruction_pairs import audit_records, generate_dataset, verify_request


class InstructionDataChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "first"
        self.counts = {"train": 3, "validation": 2, "test": 2}
        with contextlib.redirect_stdout(io.StringIO()):
            self.metadata = generate_dataset(self.output, self.counts)
        self.records = [json.loads(line) for line in (self.output / "manifest.jsonl").read_text().splitlines()]

    def test_saved_pairs_change_only_the_requested_shape(self):
        scenes = defaultdict(list)
        for record in self.records:
            scenes[record["scene_id"]].append(record)
        self.assertEqual(audit_records(self.records), self.counts)
        self.assertEqual(len(scenes), 7)
        self.assertEqual(self.metadata["total_requests"], 14)
        for requests in scenes.values():
            self.assertEqual({r["instruction_id"] for r in requests}, {0, 1})
            self.assertEqual(requests[0]["input"], requests[1]["input"])
            self.assertEqual(requests[0]["split"], requests[1]["split"])
            targets = []
            for record in requests:
                with Image.open(self.output / record["input"]) as image:
                    original = np.array(image)
                    self.assertEqual(hashlib.sha256(image.tobytes()).hexdigest(), record["input_sha256"])
                with Image.open(self.output / record["target"]) as image:
                    target = np.array(image)
                with Image.open(self.output / record["mask"]) as image:
                    selected = np.array(image) == 255
                with Image.open(self.output / record["preserve_mask"]) as image:
                    preserve = np.array(image) == 255
                self.assertTrue(np.all(original[selected] == (0, 0, 255)))
                self.assertTrue(np.all(target[selected] == (255, 0, 0)))
                np.testing.assert_array_equal(original[~selected], target[~selected])
                self.assertFalse((selected & preserve).any())
                self.assertTrue(preserve.any())
                self.assertEqual(int(np.any(original != target, axis=2).sum()), int(selected.sum()))
                self.assertIn(tuple(record["circle_tile"]), SPLIT_TILES[record["split"]])
                targets.append(target)
            self.assertFalse(np.array_equal(*targets))

    def test_same_seed_reproduces_manifest_images_and_preview(self):
        second = self.root / "second"
        with contextlib.redirect_stdout(io.StringIO()):
            generate_dataset(second, self.counts)
        paths = [self.output / "manifest.jsonl", *self.output.rglob("*.png")]
        for path in paths:
            with self.subTest(path=path.relative_to(self.output)):
                self.assertEqual(path.read_bytes(), (second / path.relative_to(self.output)).read_bytes())
        self.assertEqual(self.metadata["preview_splits"], ["train", "validation"])

    def test_scene_cannot_cross_splits(self):
        records = copy.deepcopy(self.records)
        records[1]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "cross splits"):
            audit_records(records)

    def test_input_cannot_be_reused_under_another_scene_id(self):
        records = copy.deepcopy(self.records)
        duplicates = copy.deepcopy(records[:2])
        for record in duplicates:
            record["id"] += "-duplicate"
            record["scene_id"] += "-duplicate"
        with self.assertRaisesRegex(ValueError, "different scenes"):
            audit_records(records + duplicates)

    def test_missing_or_misassigned_instruction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one request"):
            audit_records(self.records[1:])
        records = copy.deepcopy(self.records)
        records[0]["mask"] = records[0]["square_mask"]
        with self.assertRaisesRegex(ValueError, "assignment disagree"):
            audit_records(records)

    def test_corrupted_preserved_pixel_is_rejected(self):
        record = self.records[0]
        path = self.output / record["target"]
        with Image.open(path) as image:
            target = image.copy()
        with Image.open(self.output / record["preserve_mask"]) as image:
            y, x = np.argwhere(np.array(image) == 255)[0]
        target.putpixel((int(x), int(y)), (255, 0, 0))
        target.save(path)
        with self.assertRaisesRegex(ValueError, "outside the edit region"):
            verify_request(record, self.output)

    def test_existing_output_is_not_overwritten(self):
        manifest = (self.output / "manifest.jsonl").read_bytes()
        with self.assertRaises(FileExistsError):
            generate_dataset(self.output, self.counts)
        self.assertEqual((self.output / "manifest.jsonl").read_bytes(), manifest)


if __name__ == "__main__":
    unittest.main()
