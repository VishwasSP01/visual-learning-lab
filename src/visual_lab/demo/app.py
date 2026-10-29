"""Serve a local, CPU-only demo of the validation-selected spatial editor."""

import argparse
import base64
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
from visual_lab.metrics import spatial_counts, summarize_spatial
from visual_lab.models import ShapeEditor
from visual_lab.paths import ROOT
from visual_lab.rendering import render_edit
from visual_lab.thresholds import apply_threshold


def image_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class DemoEditor:
    """Load verified weights once; give the network only the selected RGB image."""

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

    def example(self, example_id):
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
            "images": {"input": image_url(original), "prediction": image_url(prediction),
                       "target": image_url(target)},
            "scores": {
                "exact_image": bool(counts["exact_images"]),
                "circle_accuracy": scores["edit_pixel_accuracy"],
                "square_preservation": scores["square_preservation"],
                "wrong_pixels": counts["pixels"] - counts["correct_pixels"],
            },
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
                self.send_json(200, {"examples": editor.ids, "threshold": editor.threshold})
            elif url.path == "/api/example":
                query = parse_qs(url.query)
                if set(query) != {"id"} or len(query["id"]) != 1:
                    self.send_json(400, {"error": "Choose one validation example."})
                    return
                try:
                    result = editor.example(query["id"][0])
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
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/shape_edits_v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/spatial_cnn/model.pt")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "docs/experiments/shape_cnn_v1_selected_config.json")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535 (0 selects an available port).")
    try:
        torch.set_num_threads(2)
        editor = DemoEditor(args.data_dir, args.checkpoint, args.config)
        server = create_server(editor, args.port)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Visual Learning Lab demo: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"{len(editor.ids)} validation examples; CPU inference; threshold {editor.threshold:.2f}.", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
