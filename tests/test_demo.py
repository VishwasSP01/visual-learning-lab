"""Check real CPU inference, label isolation, and the local demo HTTP contract."""

import base64
import hashlib
import io
import json
import random
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image

from visual_lab.data.spatial_pairs import make_pair
from visual_lab.demo.app import DemoEditor, create_server
from visual_lab.models import ShapeEditor


class DemoChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        images, _ = make_pair(random.Random(91), "validation")
        record = {"id": "validation-fixture", "split": "validation"}
        for kind, image in zip(("input", "target", "mask", "square_mask"), images, strict=True):
            record[kind] = kind + ".png"
            image.save(self.root / record[kind])
        self.circle_pixels = int((np.array(images[2]) == 255).sum())
        # Other splits deliberately have no image paths/files. They cannot be displayed.
        manifest = "\n".join(json.dumps(item) for item in (
            {"id": "train-fixture", "split": "train"}, record,
            {"id": "test-fixture", "split": "test"},
        )).encode()
        (self.root / "manifest.jsonl").write_bytes(manifest)
        self.weights = self.root / "model.pt"
        model = ShapeEditor()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.layers[-1].bias.fill_(10)
        # This test model predicts every pixel. Its errors have a hand-calculated answer.
        torch.save({"model_state": model.state_dict()}, self.weights)
        self.config = self.root / "selected.json"
        self.config.write_text(json.dumps({
            "selected_on": "validation", "test_dataset_used": False,
            "dataset": "shape_edits_v1", "model": "ShapeEditor",
            "checkpoint_sha256": hashlib.sha256(self.weights.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "mask_probability_threshold": 0.8,
        }))

    def editor(self):
        return DemoEditor(self.root, self.weights, self.config)

    def test_prediction_and_scores_are_from_saved_weights_on_cpu(self):
        editor = self.editor()
        self.assertEqual(str(next(editor.model.parameters()).device), "cpu")
        result = editor.example("validation-fixture")
        self.assertFalse(result["scores"]["exact_image"])
        self.assertEqual(result["scores"]["circle_accuracy"], 1)
        self.assertEqual(result["scores"]["square_preservation"], 0)
        self.assertEqual(result["scores"]["wrong_pixels"], 4096 - self.circle_pixels)
        raw = base64.b64decode(result["images"]["prediction"].split(",", 1)[1])
        with Image.open(io.BytesIO(raw)) as image:
            self.assertEqual(image.size, (64, 64))
            self.assertEqual(set(image.get_flattened_data()), {(255, 0, 0)})

    def test_target_changes_scores_but_cannot_change_prediction(self):
        editor = self.editor()
        before = editor.example("validation-fixture")
        Image.new("RGB", (64, 64), "white").save(self.root / "target.png")
        after = editor.example("validation-fixture")
        self.assertEqual(before["images"]["prediction"], after["images"]["prediction"])
        self.assertNotEqual(before["images"]["target"], after["images"]["target"])
        self.assertEqual(after["scores"]["circle_accuracy"], 0)

    def test_only_validation_examples_are_available(self):
        editor = self.editor()
        self.assertEqual(editor.ids, ["validation-fixture"])
        for example_id in ("train-fixture", "test-fixture", "../../model.pt"):
            with self.subTest(example_id=example_id), self.assertRaises(KeyError):
                editor.example(example_id)

    def test_modified_checkpoint_is_rejected(self):
        self.weights.write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "Checkpoint differs"):
            self.editor()

    def test_missing_checkpoint_has_an_actionable_error(self):
        self.weights.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "docs/demo.md"):
            self.editor()

    def test_http_page_and_example_contract(self):
        editor = self.editor()
        server = create_server(editor, port=0)
        self.assertEqual(server.server_address[0], "127.0.0.1")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            origin = f"http://127.0.0.1:{server.server_port}"
            with urlopen(origin, timeout=10) as response:
                page = response.read().decode()
                self.assertIn("See what the model changes.", page)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
            with urlopen(origin + "/api/examples", timeout=10) as response:
                self.assertEqual(json.load(response)["examples"], ["validation-fixture"])
            with urlopen(origin + "/api/example?id=validation-fixture", timeout=10) as response:
                result = json.load(response)
                self.assertEqual(set(result["images"]), {"input", "prediction", "target"})
                self.assertEqual(result["scores"]["wrong_pixels"], 4096 - self.circle_pixels)
            for route, code in (("/api/example?id=test-fixture", 404),
                                ("/api/example", 400), ("/LICENSE", 404)):
                with self.subTest(route=route), self.assertRaises(HTTPError) as error:
                    urlopen(origin + route, timeout=10)
                self.assertEqual(error.exception.code, code)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
