# Illustrious browser experiment

This experiment is intentionally isolated from the working FLUX.2 Klein generator.

## Target

Run an Illustrious / SDXL-family pipeline in the browser with WebGPU, with local model files and a user-supplied LoRA. The target style adapter is [Personal Ami Style - Illustrious](https://civitai.com/models/1309467/personal-ami-style-illustrious).

Known adapter metadata:
- Base family: Illustrious (SDXL-derived)
- Trigger: personal ami
- Suggested strength: 0.6–0.9
- Training base reported by the model listing: Illustrious 0.1

## Runtime decision

Do not try to bundle the Python Diffusers runtime into this static site. Diffusers depends on PyTorch and native/runtime facilities that are not directly compilable to browser WASM as-is.

Preferred architecture:

1. Use Python/Diffusers only as an offline export/conversion tool on a development machine.
2. Export the Illustrious-compatible SDXL components to ONNX with fixed, documented input/output contracts.
3. Quantize and shard the exported weights for browser delivery, then run them with the repository's vendored ONNX Runtime WebGPU runtime.
4. Keep tokenization, scheduler, image preprocessing, and orchestration in JavaScript.
5. Support local .safetensors LoRA loading in the browser. The runtime must apply compatible UNet LoRA deltas correctly; do not silently ignore unsupported LoRA tensor keys.
6. Keep model weights out of Git and load them from user-selected local files or a configurable model source.

This is a real browser-native route: inference would happen in the tab, not on a Python server. WASM CPU fallback may be useful for small components, but the denoiser should target WebGPU.

## Important implementation constraint: LoRA

A standard Diffusers LoRA file is not automatically compatible with an arbitrary exported ONNX graph. The export must preserve a way to apply the LoRA updates (for example, explicit adapter slots or a tested graph/initializer patching step). Before building the UI around a strength slider, validate one actual adapter tensor against a known Diffusers reference output.

Do not assume every LoRA is UNet-only. Inspect the file keys and report unsupported text-encoder, DoRA, LoKr, or other adapter variants explicitly.

## Why this is separate

- The working FLUX.2 Klein application remains untouched.
- Illustrious uses a different model family and pipeline.
- Its checkpoint, tokenizer, text encoders, VAE, scheduler, and LoRA format need their own validated configuration.
- The existing FLUX workers and packed-latent code must not be reused by assumption.

## Current status

Research/scaffold only. No Illustrious checkpoint or LoRA has been downloaded, no ONNX export has been produced, and browser inference is not implemented yet.

## Validation milestones

- [ ] Identify and pin an Illustrious 0.1-compatible base checkpoint with a redistributable/downloadable source.
- [ ] Inspect the exact Civitai LoRA version and tensor keys.
- [ ] Export the text encoders, UNet, and VAE to ONNX.
- [ ] Run a baseline prompt in browser WebGPU without a LoRA.
- [ ] Match a reference image from Diffusers within expected precision differences.
- [ ] Apply personal ami LoRA at strengths 0.6, 0.75, and 0.9; compare against Diffusers.
- [ ] Add arbitrary dimensions, negative prompt, CFG, sampler/steps, seed, caching, and PNG output.
- [ ] Verify memory use and generation on the target Mac/browser before calling it usable.

## References

- [ONNX Runtime WebGPU guide](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
- [Diffusers SDXL pipeline documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl)
- [Personal Ami Style - Illustrious](https://civitai.com/models/1309467/personal-ami-style-illustrious)
