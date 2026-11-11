# FLUX.1-dev experiment

A separate experiment for testing natural-language prompting with FLUX.1-dev and an adult-content-capable LoRA. This does not modify the NoobAI browser experiment.

## Initial target

- Base: `black-forest-labs/FLUX.1-dev`
- Adapter: `whrw/Flux-Uncensored-V2` (model card identifies the base as FLUX.1-dev)
- Inference: Diffusers, initially in Colab on a CUDA GPU
- Prompting: ordinary descriptive prose; no Danbooru tag syntax required
- Output: first establish a trustworthy reference image and measure memory/runtime before attempting browser deployment

## Why this is a separate experiment

FLUX.1-dev is a 12B-parameter rectified-flow transformer, not an SDXL U-Net. It uses a different text-conditioning and denoising architecture, so the NoobAI ONNX exporter, tokenizer path, scheduler, and browser worker cannot be reused as-is.

The first milestone is a known-good Diffusers reference generation with the adapter. Only after that works should we evaluate export/quantization and WebGPU feasibility.

## Access and license

The base Hugging Face repository is gated: accept its access conditions and authenticate with a Hugging Face token before downloading. FLUX.1-dev is under the FLUX.1-dev Non-Commercial License; review its terms before use. The adapter is separately published under its own model-card license. Follow both sets of terms.

## First run

Open [the Colab notebook](./FLUX_Dev_Uncensored_Reference.ipynb), select a CUDA runtime, provide a Hugging Face token with access to the base model, and run the cells. The notebook uses sequential CPU offload as a conservative memory option; it may be slow and depends on available host RAM.

This experiment is for fictional adult subjects only. Do not use it to create sexual content involving minors or non-consensual real people.

## Browser preflight

The first browser/WebGPU feasibility page is at [`browser/`](./browser/) and is deployed at:

https://dragonfiregames.github.io/imagegen/flux-experiment/browser/

Run its device report and 256/512 matrix benchmark in the target Chrome browser before attempting a model download. The benchmark is only a GPU microbenchmark, not a FLUX generation estimate.
