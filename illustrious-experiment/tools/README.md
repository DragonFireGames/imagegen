# Offline Illustrious LoRA export

This folder contains build-time tools only. The browser does not run Python,
PyTorch, Diffusers, or an inference server.

## What the exporter does

`fuse_lora_and_export.py` loads the official Diffusers pipeline
`OnomaAIResearch/Illustrious-xl-early-release-v0`, inspects the local
`.safetensors` header, loads its tensors with `safetensors.torch.load_file()`
as a state dictionary, preserves Kohya `.alpha` entries for Diffusers conversion,
verifies that UNet and both text encoders receive the adapter, and fuses it at
a chosen strength with Diffusers' safe-fusing path. It can optionally run
Optimum's SDXL ONNX exporter.

Example (use a machine or notebook with enough RAM/disk for the full SDXL model). The script defaults to BF16 to match the published Illustrious v0.1 weights; choose `--dtype float16` for hardware with limited BF16 support. `--device auto` chooses CUDA, then Apple MPS, then CPU.

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

- The user has scanned the actual Personal Ami file in the browser. Its header has
  2,958 tensors, 986 complete rank-consistent pairs, rank 32 / alpha 16 metadata,
  and all three SDXL groups: UNet, CLIP-L, and CLIP-bigG. The exporter now opens
  the local file as a Safetensors state dict (rather than passing the file path
  as a model identifier), validates the same three groups, and requires
  Diffusers to attach all of them. Full tensor loading and fusion have not yet
  been run because the raw adapter file is not present in this runtime.
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
