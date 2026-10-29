"""Serve verified categorical editing on local validation scenes, using CPU."""

import argparse
import base64
import hashlib
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import torch
from PIL import Image

from visual_lab.experiments.frozen_evaluation import validate_configuration
from visual_lab.data.instruction_pairs import audit_records
from visual_lab.experiments.instruction_training import scene_indices
from visual_lab.instruction_metrics import instruction_counts, summarize_requests
from visual_lab.metrics import spatial_counts, summarize_spatial
from visual_lab.models import ShapeEditor, SemanticDilatedEditor
from visual_lab.paths import ROOT
from visual_lab.rendering import render_edit
from visual_lab.semantic_artifacts import validate_semantic_configuration
from visual_lab.thresholds import apply_threshold


def image_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class DemoEditor:
    """Load verified weights once; give the network only the selected RGB image."""

    mode = "spatial"
    instructions = ("circle",)
    decision = "Original circle-only CNN · threshold 0.80"

    def __init__(self, data_dir, checkpoint_path, config_path):
        self.data_dir = Path(data_dir).resolve()
        for path, label in ((self.data_dir / "manifest.jsonl", "Dataset manifest"),
                            (Path(checkpoint_path), "Saved checkpoint"),
                            (Path(config_path), "Selected configuration")):
            if not path.is_file():
                raise FileNotFoundError(f"{label} missing: {path}. See docs/demo.md for prerequisites.")
        manifest = (self.data_dir / "manifest.jsonl").read_bytes()
        weights = Path(checkpoint_path).read_bytes()
        config = json.loads(Path(config_path).read_bytes())
        self.threshold = validate_configuration(config, weights, manifest)
        self.decision = f"Original circle-only CNN · threshold {self.threshold:.2f}"
        records = [json.loads(line) for line in manifest.decode().splitlines()]
        validation = [record for record in records if record["split"] == "validation"]
        self.records = {record["id"]: record for record in validation}
        if not self.records or len(self.records) != len(validation):
            raise ValueError("Expected nonempty validation data with unique example IDs.")
        self.ids = list(self.records)
        self.positions = {example_id: index + 1 for index, example_id in enumerate(self.ids)}
        self.model = ShapeEditor().cpu().eval()
        # Load the exact bytes that passed the checkpoint hash check.
        checkpoint = torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.lock = threading.Lock()

    def load_image(self, record, kind):
        path = (self.data_dir / record[kind]).resolve()
        if not path.is_relative_to(self.data_dir):
            raise ValueError("Dataset image path leaves the configured data directory.")
        with Image.open(path) as image:
            return image.copy()

    def example(self, example_id, instruction="circle"):
        if instruction not in self.instructions:
            raise KeyError("Unsupported instruction.")
        if example_id not in self.records:
            raise KeyError("Unknown validation example.")
        record = self.records[example_id]
        original = self.load_image(record, "input")
        if original.mode != "RGB" or original.size != (64, 64):
            raise ValueError("Expected a 64 x 64 RGB input.")
        inputs = torch.from_numpy(np.array(original)).permute(2, 0, 1).unsqueeze(0).float() / 255
        with self.lock, torch.inference_mode():
            logits = self.model(inputs)[0, 0].numpy()
        if not np.isfinite(logits).all():
            raise ValueError("Non-finite model output.")
        prediction = render_edit(original, apply_threshold(logits, self.threshold))
        # Labels are opened after prediction; they only supply the target panel and scores.
        target = self.load_image(record, "target")
        circle = self.load_image(record, "mask")
        square = self.load_image(record, "square_mask")
        counts = spatial_counts(original, prediction, target, circle, square)
        scores = summarize_spatial(counts)
        return {
            "id": example_id, "position": self.positions[example_id], "total": len(self.ids),
            "requested_object": "circle", "preserved_object": "square",
            "images": {"input": image_url(original), "prediction": image_url(prediction),
                       "target": image_url(target)},
            "scores": {
                "exact_image": bool(counts["exact_images"]),
                "circle_accuracy": scores["edit_pixel_accuracy"],
                "square_preservation": scores["square_preservation"],
                "edit_accuracy": scores["edit_pixel_accuracy"],
                "opposite_preservation": scores["square_preservation"],
                "wrong_pixels": counts["pixels"] - counts["correct_pixels"],
            },
        }


class SemanticDemoEditor(DemoEditor):
    """One RGB perception model, with explicit circle/square class selection."""

    mode = "semantic"
    instructions = ("circle", "square")
    decision = "Joint object prediction · choose circle or square"
    threshold = None

    def __init__(self, data_dir, checkpoint_path, config_path):
        self.data_dir = Path(data_dir).resolve()
        for path, label in ((self.data_dir / "manifest.jsonl", "Dataset manifest"),
                            (Path(checkpoint_path), "Saved checkpoint"), (Path(config_path), "Selected configuration")):
            if not path.is_file():
                raise FileNotFoundError(f"{label} missing: {path}. See docs/demo.md for prerequisites.")
        manifest = (self.data_dir / "manifest.jsonl").read_bytes()
        weights = Path(checkpoint_path).read_bytes()
        config = json.loads(Path(config_path).read_bytes())
        validate_semantic_configuration(config, weights, manifest)
        records = [json.loads(line) for line in manifest.decode().splitlines()]
        audit_records(records)  # Metadata only; no train or test image is opened.
        selected = [r for r in records if r["split"] == "validation"]
        groups = scene_indices(selected)
        self.records = {scene: {task: selected[index] for task, index in pair.items()} for scene, pair in groups.items()}
        self.ids = list(self.records)
        self.positions = {scene: i + 1 for i, scene in enumerate(self.ids)}
        self.model = SemanticDilatedEditor().cpu().eval()
        self.model.load_state_dict(torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True)["model_state"], strict=True)
        self.lock = threading.Lock()

    def example(self, example_id, instruction="circle"):
        if example_id not in self.records or instruction not in self.instructions:
            raise KeyError("Unknown validation scene or instruction.")
        record = self.records[example_id][self.instructions.index(instruction)]
        original = self.load_image(record, "input")
        if original.mode != "RGB" or original.size != (64, 64):
            raise ValueError("Expected a 64 x 64 RGB input.")
        if hashlib.sha256(original.tobytes()).hexdigest() != record["input_sha256"]:
            raise ValueError("Input pixels differ from the selected manifest.")
        inputs = torch.from_numpy(np.array(original)).permute(2, 0, 1).unsqueeze(0).float() / 255
        with self.lock, torch.inference_mode():
            logits = self.model(inputs)
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite model output.")
            classes = logits.argmax(dim=1)[0].numpy()
        prediction = render_edit(original, classes == record["instruction_id"] + 1)
        # Open labels only after predicting the complete class map from RGB.
        target = self.load_image(record, "target")
        counts = instruction_counts(original, prediction, target, self.load_image(record, "mask"),
                                    self.load_image(record, "preserve_mask"))
        scores = summarize_requests(counts)
        return {
            "id": example_id, "request_id": record["id"], "position": self.positions[example_id], "total": len(self.ids),
            "requested_object": instruction, "preserved_object": "square" if instruction == "circle" else "circle",
            "images": {"input": image_url(original), "prediction": image_url(prediction), "target": image_url(target)},
            "scores": {"exact_image": bool(counts["exact_images"]), "edit_accuracy": scores["edit_pixel_accuracy"],
                       "opposite_preservation": scores["opposite_object_preservation"],
                       "wrong_pixels": counts["pixels"] - counts["correct_pixels"]},
        }


def create_server(editor, port=7860):
    page = Path(__file__).with_name("index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def send_content(self, code, body, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, code, value):
            self.send_content(code, json.dumps(value).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/":
                self.send_content(200, page, "text/html; charset=utf-8")
            elif url.path == "/api/examples":
                self.send_json(200, {"examples": editor.ids, "threshold": editor.threshold, "mode": editor.mode,
                                     "instructions": editor.instructions, "decision": editor.decision})
            elif url.path == "/api/example":
                query = parse_qs(url.query)
                if ("id" not in query or not set(query) <= {"id", "instruction"} or len(query["id"]) != 1
                        or ("instruction" in query and len(query["instruction"]) != 1)):
                    self.send_json(400, {"error": "Choose one validation example."})
                    return
                instruction = query.get("instruction", ["circle"])[0]
                if instruction not in editor.instructions:
                    self.send_json(400, {"error": "Choose an available circle or square instruction."})
                    return
                try:
                    result = editor.example(query["id"][0], instruction)
                except KeyError:
                    self.send_json(404, {"error": "Unknown validation example."})
                except (OSError, ValueError, RuntimeError):
                    self.send_json(500, {"error": "Could not load this example. Check the local data files."})
                else:
                    self.send_json(200, result)
            else:
                self.send_json(404, {"error": "Not found."})

        def log_message(self, format, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("semantic", "spatial"), default="semantic")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535 (0 selects an available port).")
    try:
        torch.set_num_threads(2)
        semantic = args.mode == "semantic"
        data_dir = args.data_dir or ROOT / ("data/instruction_edits_v1" if semantic else "data/shape_edits_v1")
        checkpoint = args.checkpoint or ROOT / ("results/reproductions/semantic_dilated_v2/model.pt" if semantic else "results/spatial_cnn/model.pt")
        config = args.config or ROOT / ("docs/experiments/semantic_dilated_v2_selected_config.json" if semantic else "docs/experiments/shape_cnn_v1_selected_config.json")
        constructor = SemanticDemoEditor if semantic else DemoEditor
        editor = constructor(data_dir, checkpoint, config)
        server = create_server(editor, args.port)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Visual Learning Lab demo: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"{len(editor.ids)} validation scenes; CPU inference; {editor.decision}.", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
