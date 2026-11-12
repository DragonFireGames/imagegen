# Offline SDXL LoRA export

This folder contains build-time tools only. The browser generator does not run
Python, PyTorch, Diffusers, or an inference server.

## What the exporter does

`fuse_lora_and_export.py` loads the configured Illustrious Diffusers pipeline,
reads a local `.safetensors` LoRA state dictionary, checks its supported SDXL
component groups, attaches the adapter through Diffusers, and fuses the selected
strength using safe-fusing. It can optionally run Optimum's SDXL ONNX exporter.

The adapter may target any supported subset of the SDXL UNet, CLIP-L, and
CLIP-bigG components. The script does not require all three groups to be present.

## Example

Use a machine or notebook with sufficient RAM and disk for the full SDXL base
model. The default base is Illustrious XL v0.1. The default fusion strength is
1.0; set it explicitly to compare different scales.

```bash
# Install a PyTorch build for your OS/GPU first:
# https://pytorch.org/get-started/locally/

python -m venv .venv
source .venv/bin/activate
python -m pip install -r illustrious-experiment/tools/requirements-export.txt

python illustrious-experiment/tools/fuse_lora_and_export.py \
  --lora /path/to/your-sdxl-lora.safetensors \
  --strength 1.0 \
  --dtype bfloat16 \
  --device auto \
  --output /path/to/fused-sdxl-lora \
  --export-onnx
```

On Windows, activate with `.venv\\Scripts\\activate`.

Omit `--export-onnx` to save a fused Diffusers pipeline instead. Add
`--keep-diffusers` when exporting ONNX to retain the merged Diffusers
intermediate in `OUTPUT/merged-diffusers`. Choose `--dtype float16` on
hardware without reliable BF16 support.

## Compatibility and limitations

- The exporter accepts standard SDXL Kohya-style LoRA groups for UNet,
  CLIP-L, and CLIP-bigG. A file may contain only a subset of these groups.
- It validates paired down/up or A/B tensor shapes and component coverage
  before loading the full pipeline. Unsupported key layouts or extended
  adapter types are rejected rather than silently ignored.
- A successful Diffusers fusion does not prove that the exported ONNX model is
  compatible with the browser's split WebGPU graphs.
- The browser patcher targets this repository's WAI Illustrious v17 SDXL ONNX
  weights index. It supports compatible linear and Conv2D down/up factors in
  FP16, BF16, or FP32, with per-layer alpha or the adapter metadata alpha
  fallback. Shape and target-index checks are enforced.
- Adapter files remain local user-supplied files. They are not uploaded by the
  browser tools or committed to this repository.
- Fused model exports are multi-gigabyte artifacts. Store them outside this
  repository; they are not meant to be committed to Git.
- The configured Illustrious base is distributed under Fair AI Public License
  1.0-SD. Read https://freedevproject.org/faipl-1.0-sd/ and the base model's
  terms before publishing a fused derivative.

## Why this is useful

It provides a local Diffusers reference for checking a base model plus a chosen
LoRA and strength. Use the same prompt, seed, sampler, resolution, and other
settings when comparing it with the browser implementation.
