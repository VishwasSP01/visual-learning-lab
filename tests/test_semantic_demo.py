"""Check categorical demo choices, RGB-only prediction, and validation isolation."""

import base64
import hashlib
import io
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image

from visual_lab.data.instruction_pairs import generate_dataset
from visual_lab.demo.app import SemanticDemoEditor, create_server
from visual_lab.models import SemanticDilatedEditor


class SemanticDemoChecks(unittest.TestCase):
    def setUp(self):
        import contextlib
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.data = self.root / "data"
        with contextlib.redirect_stdout(io.StringIO()):
            generate_dataset(self.data, {"train": 1, "validation": 1, "test": 1})
        for split in ("train", "test"):shutil.rmtree(self.data / split)
        manifest = (self.data / "manifest.jsonl").read_bytes()
        self.records = [json.loads(line) for line in manifest.decode().splitlines() if json.loads(line)["split"] == "validation"]
        self.scene = self.records[0]["scene_id"]
        self.weights = self.root / "model.pt"
        model = SemanticDilatedEditor()
        with torch.no_grad():
            for p in model.parameters():p.zero_()
            model.layers[-1].bias[1] = 10
        torch.save({"model_state": model.state_dict()}, self.weights)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"model": "SemanticDilatedEditor", "architecture": "dilated",
                                          "dataset": "instruction_edits_v1", "decoder": "three_class_argmax",
                                          "class_mapping": {"background": 0, "circle": 1, "square": 2},
                                          "selected_on": "validation", "test_model_performance_measured": False,
                                          "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                                          "checkpoint_sha256": hashlib.sha256(self.weights.read_bytes()).hexdigest()}))

    def editor(self):
        return SemanticDemoEditor(self.data, self.weights, self.config)

    def test_choices_use_same_input_and_predictions_have_hand_calculated_errors(self):
        editor = self.editor()
        circle, square = editor.example(self.scene, "circle"), editor.example(self.scene, "square")
        self.assertEqual(str(next(editor.model.parameters()).device), "cpu")
        self.assertEqual(circle["images"]["input"], square["images"]["input"])
        self.assertNotEqual(circle["images"]["prediction"], square["images"]["prediction"])
        with Image.open(self.data / self.records[0]["mask"]) as image:circle_pixels = int((np.array(image) == 255).sum())
        with Image.open(self.data / self.records[1]["mask"]) as image:square_pixels = int((np.array(image) == 255).sum())
        self.assertEqual(circle["scores"]["wrong_pixels"], 4096 - circle_pixels)
        self.assertEqual(circle["scores"]["edit_accuracy"], 1)
        self.assertEqual(circle["scores"]["opposite_preservation"], 0)
        self.assertEqual(square["scores"]["wrong_pixels"], square_pixels)
        self.assertEqual(square["scores"]["edit_accuracy"], 0)
        self.assertEqual(square["scores"]["opposite_preservation"], 1)
        self.assertEqual(square["preserved_object"], "circle")
        with Image.open(io.BytesIO(base64.b64decode(circle["images"]["prediction"].split(",", 1)[1]))) as image:
            self.assertEqual(set(image.get_flattened_data()), {(255, 0, 0)})

    def test_changing_labels_cannot_change_rgb_prediction(self):
        editor = self.editor(); before = editor.example(self.scene, "circle")
        Image.new("RGB", (64, 64), "white").save(self.data / self.records[0]["target"])
        after = editor.example(self.scene, "circle")
        self.assertEqual(before["images"]["prediction"], after["images"]["prediction"])
        self.assertNotEqual(before["images"]["target"], after["images"]["target"])
        self.assertEqual(after["scores"]["edit_accuracy"], 0)

    def test_only_validation_scenes_and_two_known_choices_are_available(self):
        editor = self.editor(); self.assertEqual(editor.ids, [self.scene])
        for scene, instruction in (("test-scene-0000", "circle"), ("train-scene-0000", "circle"),
                                    ("../../model.pt", "circle"), (self.scene, "triangle")):
            with self.assertRaises(KeyError):editor.example(scene, instruction)

    def test_changed_weights_and_manifest_are_rejected(self):
        original = self.weights.read_bytes(); self.weights.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Checkpoint differs"):self.editor()
        self.weights.write_bytes(original)
        with (self.data / "manifest.jsonl").open("ab") as manifest:manifest.write(b"\n")
        with self.assertRaisesRegex(ValueError, "Manifest differs"):self.editor()

    def test_http_instruction_choices_and_validation_only_contract(self):
        editor = self.editor(); server = create_server(editor, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            origin = f"http://127.0.0.1:{server.server_port}"
            with urlopen(origin + "/api/examples", timeout=10) as response:
                metadata = json.load(response)
            self.assertEqual(metadata["instructions"], ["circle", "square"])
            self.assertEqual(metadata["mode"], "semantic")
            for instruction in metadata["instructions"]:
                with urlopen(origin + f"/api/example?id={self.scene}&instruction={instruction}", timeout=10) as response:
                    result = json.load(response)
                self.assertEqual(result["requested_object"], instruction)
            for route, code in ((f"/api/example?id={self.scene}&instruction=triangle", 400),
                                ("/api/example?id=test-scene-0000&instruction=square", 404),
                                (f"/api/example?id={self.scene}&instruction=circle&instruction=square", 400)):
                with self.assertRaises(HTTPError) as error:urlopen(origin + route, timeout=10)
                self.assertEqual(error.exception.code, code)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)
