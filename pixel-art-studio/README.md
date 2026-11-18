# Pixel Foundry — local AI pixel-art generator

The previous pixel editor has been removed. This is an AI-only sprite generator: no drawing canvas, brush tools, layers, or Illustrious model.

## Model

[Sprite32 1.0 fast](https://huggingface.co/danemarguglio/sprite32) is a text-to-sprite diffusion model trained from scratch on 660,585 images. It generates native 32×32 RGBA sprites with transparency using a 143M-parameter pixel-space UNet. The fast release uses a fine-tuned Qwen3-0.6B text encoder and a distilled 12-step sampler.

- About 2.3 GB of model files including the bundled text encoder.
- Apache-2.0.
- Downloaded and run locally; no API key or hosted inference.
- Fixed 32×32 output, intended for icons, objects, characters and game sprites.

## Run

macOS: double-click start.command, or run ./start.sh in Terminal.
Linux: run ./start.sh.
Requirements: Python 3.11+ and Git. First launch clones the upstream inference code, installs dependencies, and downloads weights. Then open http://127.0.0.1:8077.

The model author's fast checkpoint has been tested on Linux CPU/GPU. Apple Silicon device selection is supported in the upstream code, but the fast checkpoint's MPS path has not been independently verified. The full checkpoint is the author's documented Apple-Silicon-tested alternative, but requires about 8.7 GB.

## Honest capability boundary

Current: text-to-sprite generation, batches, random/fixed seeds, prompt-strength control, local history and transparent PNG export.

Not yet implemented: editing an uploaded sprite, masked inpainting, pose transfer, animation, or guaranteed character identity. Sprite32 is a text-to-image model, not an image-editing model. These features need a separate model trained for image conditioning; this project does not fake them with pixel filters.

The upstream model code is cloned into .runtime/sprite32 at first launch. We reuse its sampler and local FastAPI server instead of duplicating the model implementation.
