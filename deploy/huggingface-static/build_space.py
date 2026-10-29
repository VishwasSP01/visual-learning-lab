"""Build a self-contained Static Space from the verified frozen checkpoint."""

import argparse
import base64
import hashlib
import io
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from visual_lab.data.loaders import load_instruction_split
from visual_lab.experiments.instruction_training import scene_indices
from visual_lab.models import SemanticDilatedEditor
from visual_lab.paths import ROOT
from visual_lab.provenance import git_state
from visual_lab.semantic_artifacts import validate_semantic_configuration

RUNTIME_VERSION = "1.31.1"
RUNTIME_INTEGRITY = "5GeH+bWQ+HxM10z7erGlsWqryrwMbSvm/1tkIsKTT48g0fI19VXVVC435t4t7VsPdSQeS7NmuZPXrQ73k82URw=="


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/huggingface-static")
    parser.add_argument("--runtime-tar", type=Path, default=ROOT / f"results/onnxruntime-web-{RUNTIME_VERSION}.tgz")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    data_dir = ROOT / "data/instruction_edits_v1"
    config_bytes = (ROOT / "docs/experiments/semantic_dilated_v2_selected_config.json").read_bytes()
    config = json.loads(config_bytes)
    checkpoint_bytes = (ROOT / "results/reproductions/semantic_dilated_v2/model.pt").read_bytes()
    manifest = (data_dir / "manifest.jsonl").read_bytes()
    validate_semantic_configuration(config, checkpoint_bytes, manifest)
    model = SemanticDilatedEditor().cpu().eval()
    model.load_state_dict(torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)["model_state"], strict=True)
    torch.set_num_threads(1)
    model_path = args.output / "model.onnx"
    torch.onnx.export(model, (torch.zeros(1, 3, 64, 64),), str(model_path),
                      input_names=["images"], output_names=["logits"],
                      dynamo=False, opset_version=18, external_data=False)
    onnx.checker.check_model(onnx.load(model_path))
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    records = [json.loads(line) for line in manifest.decode().splitlines()]
    data = load_instruction_split(data_dir, records, "validation")
    examples, expected, maximum_error = [], {}, 0.0
    all_reference, all_exported = [], []
    for scene_id, pair in scene_indices(data[3]).items():
        index = pair[0]
        pixels = data[0][index].numpy()
        inputs = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None]).astype(np.float32) / 255
        with torch.inference_mode(): reference = model(torch.from_numpy(inputs)).numpy()
        exported = session.run(["logits"], {"images": inputs})[0]
        maximum_error = max(maximum_error, float(np.abs(reference - exported).max()))
        classes, converted_classes = reference.argmax(1)[0], exported.argmax(1)[0]
        if not np.array_equal(classes, converted_classes):
            raise RuntimeError(f"Export changed a validation class prediction: {scene_id}")
        all_reference.append(classes.astype(np.uint8)); all_exported.append(converted_classes.astype(np.uint8))
        record = data[3][index]
        row = {"id": scene_id, "input": "data:image/png;base64," + base64.b64encode((data_dir / record["input"]).read_bytes()).decode()}
        expected[scene_id] = {}
        for task, task_id in (("circle", 0), ("square", 1)):
            mask = data[2][pair[task_id]].numpy()
            row[task + "_mask"] = base64.b64encode(np.packbits(mask).tobytes()).decode()
            prediction = np.concatenate((pixels.copy(), np.full((64, 64, 1), 255, np.uint8)), axis=2)
            prediction[classes == task_id + 1] = (255, 0, 0, 255)
            expected[scene_id][task] = {"rgba_sha256": sha(prediction.tobytes()),
                                      "wrong_pixels": int(((classes == task_id + 1) != mask.astype(bool)).sum())}
        examples.append(row)
    examples_bytes = (json.dumps(examples, separators=(",", ":")) + "\n").encode()
    (args.output / "examples.json").write_bytes(examples_bytes)
    (args.output / "checkpoint.pt").write_bytes(checkpoint_bytes)
    (args.output / "selected_config.json").write_bytes(config_bytes)
    for name in ("README.md", "index.html", "app.mjs", "inference.mjs"):
        shutil.copyfile(source / name, args.output / name)
    shutil.copyfile(ROOT / "LICENSE", args.output / "LICENSE")
    if not args.runtime_tar.is_file():
        url = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{RUNTIME_VERSION}.tgz"
        with urllib.request.urlopen(url, timeout=120) as response: raw = response.read()
        args.runtime_tar.parent.mkdir(parents=True, exist_ok=True); args.runtime_tar.write_bytes(raw)
    raw = args.runtime_tar.read_bytes()
    if base64.b64encode(hashlib.sha512(raw).digest()).decode() != RUNTIME_INTEGRITY:
        raise ValueError("ONNX Runtime package integrity mismatch.")
    vendor = args.output / "vendor"; vendor.mkdir()
    runtime_hashes = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for name in ("ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"):
            content = archive.extractfile("package/dist/" + name).read()
            (vendor / name).write_bytes(content); runtime_hashes[name] = sha(content)
    # The npm distribution omits these files; include the pinned upstream notices.
    licenses = {"LICENSE": "2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c",
                "ThirdPartyNotices.txt": "143764b952fdb1a7c69ce653bfba74a7744d6a8a573bfb73e235fba356c83de3"}
    for name, expected in licenses.items():
        url = f"https://raw.githubusercontent.com/microsoft/onnxruntime/v{RUNTIME_VERSION}/{name}"
        with urllib.request.urlopen(url, timeout=30) as response: content = response.read()
        if sha(content) != expected: raise ValueError(f"Upstream notice checksum mismatch: {name}")
        (vendor / name).write_bytes(content); runtime_hashes[name] = sha(content)
    metadata = {
        **git_state(), "checkpoint_sha256": config["checkpoint_sha256"], "checkpoint_bytes": len(checkpoint_bytes),
        "onnx_sha256": sha(model_path.read_bytes()), "onnx_bytes": model_path.stat().st_size,
        "examples_sha256": sha(examples_bytes), "validation_input_mask_sha256": data[4],
        "runtime": "onnxruntime-web", "runtime_version": RUNTIME_VERSION,
        "runtime_package_sha512_base64": RUNTIME_INTEGRITY, "runtime_assets_sha256": runtime_hashes,
        "torch_version": str(torch.__version__), "onnx_version": onnx.__version__, "onnxruntime_version": ort.__version__,
        "opset": 18, "input": "float32 NCHW [1,3,64,64] RGB /255", "output": "3 class logits, lowest-index argmax",
        "native_onnx_class_agreement": True, "validation_scenes_compared": len(examples),
        "validation_requests_covered": 2 * len(examples), "max_absolute_logit_difference": maximum_error,
        "pytorch_class_maps_sha256": sha(np.stack(all_reference).tobytes()),
        "onnx_class_maps_sha256": sha(np.stack(all_exported).tobytes()),
        "source_files_sha256": {name: sha((source / name).read_bytes()) for name in
                                ("build_space.py", "README.md", "index.html", "app.mjs", "inference.mjs", "requirements-export.txt")},
        "training_performed": False, "test_images_loaded": False,
        "scope": "Format conversion and deployment parity on known validation scenes. Frozen test score belongs to original PyTorch evaluation.",
    }
    (args.output / "model_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (args.output.parent / (args.output.name + "-expected.json")).write_text(json.dumps(expected))
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Upload-ready Static Space: {args.output}")


if __name__ == "__main__": main()
