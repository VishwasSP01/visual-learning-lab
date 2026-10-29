# Free Hugging Face Static Space

## Deployment status

The upload-ready app has been built and tested locally. Publishing is waiting
for the owner to create/sign into Hugging Face and complete email verification.
No hosted Space URL is claimed or linked from the README until the Space exists
and its app has been verified live.

The deployment sources and parity reports are versioned in GitHub. The original
checkpoint and generated upload bundle stay in the ignored `results/` folder.

The user selected **a free Static Space with browser inference** after checking
the current hosting rules. The [current Spaces overview](https://huggingface.co/docs/hub/spaces-overview)
requires a paid plan for new compute-backed CPU Gradio Spaces. Free ZeroGPU
hosting requires a verified personal account older than 30 days. In contrast,
[Static Spaces](https://huggingface.co/docs/hub/spaces-sdks-static) are free for
new accounts and serve files without a Python inference server.

## What is packaged

Source: `deploy/huggingface-static/`. Generated upload bundle:
`results/huggingface-static/` (excluded from Git).

- Static HTML and JavaScript with circle/square choices, all 250 validation
  scenes, original/prediction/target panels, and per-request scores.
- Original 33,445-byte checkpoint, bound to SHA-256
  `e114687642887e771d69684b9723ae19ffdf7200cedafc202a99de483894e90d`.
- A 31,665-byte ONNX export with the same float32 weights and three-class argmax.
- Pinned ONNX Runtime Web **1.31.1**, including locally served JavaScript/WASM
  assets, license, and third-party notices included in the package.
- Export metadata, selected configuration, validation examples, and project license.

The runtime assets total about 15 MB; the model itself is about 32 KB. Browser
latency varies by device and differs from the original sub-millisecond Python
CPU measurement. The app displays measured browser inference time. The frozen
97.2% test result remains attributed to the original PyTorch evaluation.

Only RGB enters the inference session. The instruction selects a predicted
class; ground-truth masks supply the target and scores. Targets never enter
the network. The app uses known synthetic validation scenes, fixed categories,
and a fixed red renderer. It does not perform arbitrary photo editing.

## Build without changing the research environment

From the repository root, with the original local artifacts available:

```bash
uv venv --python 3.12 results/space-export-env
uv pip install --python results/space-export-env/bin/python -e . -r deploy/huggingface-static/requirements-export.txt
results/space-export-env/bin/python deploy/huggingface-static/build_space.py
uv run --locked python -m http.server 7861 --bind 127.0.0.1 --directory results/huggingface-static
```

The builder requires a new output directory; pass `--output` for another build.
It verifies the selected checkpoint/manifest, exports a fixed 64x64 RGB model at
ONNX opset 18, compares all validation scene predictions, and checks the pinned
runtime package's SHA-512 before extracting only named assets. The original
checkpoint, root dependencies, and `uv.lock` remain unchanged. Test images are
never loaded for export or parity checking.

## Verification

- Native ONNX Runtime matches every class-map pixel on 250 validation scenes.
  Maximum absolute logit difference: approximately `2.48e-5`; argmax unchanged.
- The actual local browser was exercised under both choices on all 250 scenes.
  Every rendered PNG's decoded RGBA bytes match the corresponding frozen-model
  prediction, and every displayed wrong-pixel count matches independently.
- Browser results reproduce **491/500 exact requests**, **244/250 exact pairs**,
  and **20 wrong pixels** on validation. This is deployment parity, not new
  unseen evidence.
- Three Node fixture tests check RGB-to-NCHW normalization, argmax/ties,
  non-finite rejection, packed-mask bit order, preservation scoring, and painting.

Reports: [native export](experiments/static_space_export.json) and
[browser audit](experiments/static_space_browser_audit.json).

```bash
node --test deploy/huggingface-static/inference.test.mjs
```

## Publish after account setup

Create a public Hugging Face Space using **Static** as the SDK. Upload the
contents of the generated bundle to the Space repository root, including its
YAML-front-matter `README.md`, model, checkpoint, examples, and `vendor/` directory.
No paid hardware or Gradio SDK is used. Confirm both requests work in the hosted
app, then place the verified Space URL near the top of the GitHub README.

Official runtime references: [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
and [deployment assets](https://onnxruntime.ai/docs/tutorials/web/deploy.html).
