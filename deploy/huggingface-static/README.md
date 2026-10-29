---
title: Visual Learning Lab
emoji: 🔵
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
license: mit
short_description: Choose circle or square; frozen image editing in your browser
---

# Visual Learning Lab

A 7,459-parameter semantic editor, exported from the frozen seed-42 checkpoint.
Choose which object becomes red while preserving the other object and background.

Inference runs locally in the visitor's browser with ONNX Runtime Web/WASM.
The Space serves static files and needs no paid inference hardware. Browser
latency varies by device; the reported sub-millisecond Python CPU benchmark is
not a browser latency guarantee.

The original PyTorch model achieved 972/1,000 exact requests and 482/500 exact
scene pairs on the frozen synthetic test. This app displays validation scenes.
Targets and ground-truth masks supply comparison and scoring, never model inputs.
Synthetic shapes and fixed instruction categories; not real-photo editing.

[Source and experiments](https://github.com/VishwasSP01/visual-learning-lab)
· [Frozen test report](https://github.com/VishwasSP01/visual-learning-lab/blob/main/docs/instruction_test_evaluation.md)

Files include the verified original checkpoint (`checkpoint.pt`), its selected
config, the browser model (`model.onnx`), export/parity metadata, validation
examples, and the locally served pinned ONNX Runtime distribution and license.
