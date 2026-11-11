# FLUX.1-dev browser experiment

This is a separate browser/WebGPU experiment for the same FLUX.1-dev + `whrw/Flux-Uncensored-V2` reference setup. It does not modify the existing FLUX.2 Klein app or NoobAI XL browser experiment.

## Browser preflight

Open the GitHub Pages page:

`https://dragonfiregames.github.io/imagegen/flux-experiment/browser/`

It:
- Requests the user's high-performance WebGPU adapter.
- Reports WebGPU limits that matter for large transformer tensors.
- Checks optional `shader-f16` support.
- Runs 256×256 and 512×512 GPU matrix multiplication microbenchmarks, reporting median latency and estimated GFLOP/s.
- Offers a copyable diagnostic report.

The microbenchmark is deliberately not presented as a FLUX generation benchmark. It is a hardware/kernel baseline only. WebGPU does not expose total or free VRAM to page JavaScript, so the page reports API limits, not invented memory numbers.

## Architecture direction

1. Browser device preflight.
2. Inspect the official gated `black-forest-labs/FLUX.1-dev-onnx` graph files and their operators/data types.
3. Validate one real FLUX transformer block in the browser against the Colab/PyTorch reference.
4. Add text encoders, flow-matching scheduler, VAE, and the uncensored LoRA (fused into transformer weights or applied separately).
5. Only then benchmark full text-to-image performance.

Black Forest Labs publishes FLUX.1-dev ONNX exports in BF16, FP8, and FP4. The repository is gated; users must accept its conditions. ONNX Runtime WebGPU documents operator support, including `MatMulNBits`, but actual model graph compatibility and performance must be tested in Chrome before choosing this path.

## Important

- Target model is FLUX.1-dev (12B rectified-flow transformer), not FLUX.2 Klein 4B.
- The reference image was generated with the FLUX.1-dev base plus `whrw/Flux-Uncensored-V2`.
- Do not copy FLUX.2 Klein's architecture-specific model exporter or scheduler into this experiment.
- No model weights are downloaded by the preflight page.
- FLUX.1-dev is under its Non-Commercial License; review the base and adapter licenses before use.
