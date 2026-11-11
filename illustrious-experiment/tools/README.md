# Offline Illustrious LoRA export

This folder contains build-time tools only. The browser does not run Python,
PyTorch, Diffusers, or an inference server.

## What the exporter does

`fuse_lora_and_export.py` loads the official Diffusers pipeline
`OnomaAIResearch/Illustrious-xl-early-release-v0`, loads a local SDXL
`.safetensors` adapter, fuses it into the model at a chosen strength using
Diffusers' safe-fusing path, and optionally runs Optimum's SDXL ONNX exporter.

Example (use a machine or notebook with enough RAM/disk for the full SDXL model):

```bash
# Install a PyTorch build for your OS/GPU first:
# https://pytorch.org/get-started/locally/

python -m venv .venv
source .venv/bin/activate
python -m pip install -r illustrious-experiment/tools/requirements-export.txt

python illustrious-experiment/tools/fuse_lora_and_export.py \
  --lora /path/to/personal-ami.safetensors \
  --strength 0.75 \
  --output /path/to/personal-ami-onnx \
  --export-onnx
```

On Windows, activate with `.venv\\Scripts\\activate`.

Omit `--export-onnx` to save a fused Diffusers pipeline instead. Add
`--keep-diffusers` when exporting ONNX to retain the merged Diffusers
intermediate in `OUTPUT/merged-diffusers`.

## Important limits

- This script has not been run against the Personal Ami file yet; that file is
  not in this repository. A successful Diffusers load and fuse is the first
  compatibility check.
- The Optimum export is a **reference ONNX export**, not the optimized,
  split WebGPU package used by BLIP Studio. Do not point the browser at it
  until its graph operators, external-data layout, numerical output, and
  memory behavior have been tested in ONNX Runtime WebGPU.
- A fused model has a fixed LoRA strength. It does not support live slider
  changes or arbitrary adapter hot-swapping.
- Model output files are multi-gigabyte artifacts. Store them outside this
  repository; they are not meant to be committed to Git.
- The official Illustrious base is distributed under Fair AI Public License
  1.0-SD. Read its license and terms before using or publishing merged
  derivatives: https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0
- The Personal Ami adapter's own license must be checked at its original
  Civitai model/version page. The base-model license does not grant rights to
  the adapter.

## Why this is useful

It gives us a deterministic Diffusers reference using the exact requested
base model and adapter strength. That reference can validate the browser
implementation later, instead of guessing whether an ONNX output is correct.
