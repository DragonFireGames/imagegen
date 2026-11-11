# Illustrious browser experiment

This experiment is intentionally isolated from the working FLUX.2 Klein generator.

## Target adapter

[Personal Ami Style - Illustrious](https://civitai.com/models/1309467/personal-ami-style-illustrious) is listed as an Illustrious LoRA trained on Illustrious XL 0.1. Its mirrored listing gives the trigger personal ami and suggests a strength of 0.6–0.9. The mirror identifies its own copy as a reprint, so the original Civitai page remains the source of record.

## Browser-native runtime candidates

### Option A: pre-exported ONNX + ONNX Runtime WebGPU

A compatible SDXL-derived ONNX/WebGPU model pack already exists:

- [WAI Illustrious SDXL v17 ONNX package](https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17)
- Its model card declares its base as [OnomaAIResearch/Illustrious-xl-early-release-v0](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0), the Illustrious XL v0.1 base used by the target LoRA.
- The package includes two CLIP text encoders, two tokenizers, a three-part SDXL UNet, VAE encoder/decoder, and weights-index.json files that its publisher says are used to apply LoRAs in the browser.
- The full repository is 8.75 GB because it includes both FP16 and INT4 graph variants. The FP16 UNet graphs alone are about 5.13 GB; text encoders are about 1.62 GB and VAE graphs about 0.17 GB.
- The INT4 variants reduce the two largest UNet segments to about 1.82 GB combined, but the model manifest does not declare weights-index files for those INT4 graphs. Therefore they are not the initial LoRA target.

This is a promising runtime starting point, not a claim of tested compatibility. It is WAI Illustrious v17, a fine-tune derived from Illustrious v0.1—not the untouched v0.1 checkpoint. We still need to apply the Personal Ami adapter and compare output against the same LoRA loaded with Diffusers.

The ONNX pack is under Fair AI Public License 1.0-SD, which contains prohibited-use and source/license notice requirements. Review the pack's LICENSE.txt before use or redistribution. The original Illustrious v0.1 repository also lists that license and provides a TERM_OF_USE.

### Option B: ExecuTorch WebGPU export

As of November 10, 2026, PyTorch has announced an experimental ExecuTorch WebGPU backend. It accepts a PyTorch export and packages it as an ExecuTorch .pte program for the browser's WebGPU runtime. That is closer to the user's preferred "compile Python to WebAssembly/JavaScript" direction: Python is used for export, while the browser runs the exported program through a WebAssembly runtime and WebGPU.

This is not yet a proven SDXL route:

- The current backend is explicitly experimental.
- Its documentation describes more than 100 operators but stresses that support depends on shape, layout and dtype; unsupported operators may need CPU fallback or additional kernels.
- The documented minimal export example delegates an add operation, and the current docs do not demonstrate an SDXL/Illustrious UNet or full Stable Diffusion XL pipeline on this backend.
- Stable Diffusion requires separate text encoders, UNet, VAE, scheduler, and preprocessing. Each component must export and execute correctly.
- Dynamic LoRA loading would need an explicit adapter strategy (for example, adapter weights as model inputs or graph variants). A .pte export does not automatically make a .safetensors LoRA dynamically loadable.

We should prototype the difficult SDXL UNet operations with ExecuTorch before switching away from the pre-exported ONNX candidate. It may ultimately be cleaner than ONNX, but there is not enough evidence yet to claim it works for this pipeline.

### Option C: adapt an existing SDXL browser pipeline

The [d0grr/generate-ai-images](https://github.com/d0grr/generate-ai-images) project provides another concrete reference: it runs SDXL in Chrome through ONNX Runtime WebGPU, uses a JavaScript SDXL pipeline with dual CLIP encoders, scheduler and VAE, caches weights in OPFS, and offers FP16 and INT4 UNet model variants. Its author documents roughly 8 GB RAM and 8 GB VRAM as the recommended full-SDXL target, with a smaller 4-bit UNet for constrained GPUs.

This helps answer the feasibility question: browser-native SDXL is already demonstrated. The project’s documented custom SDXL inpainting LoRA is merged into the UNet during its Python export; it is not an example of live, arbitrary Safetensors adapter swapping. We can study its pipeline and low-memory techniques, but should not copy code until its license and the exact components we need are checked.

For this project, the intended sequence is to export an Illustrious UNet with the Personal Ami adapter merged at a chosen strength as the first working browser model, then add runtime LoRA swapping using the WAI pack's weight-index approach if the exact tensor mapping validates. That gives us a path to a working model without pretending live adapter injection already works.

## Runtime decision

Do not try to bundle the Python Diffusers runtime into this static site. Diffusers depends on PyTorch and native/runtime facilities that are not directly compilable to browser WASM as-is.

Preferred sequence:

1. First try the pre-exported FP16 ONNX graphs because they already exist and have weight-index metadata for browser LoRA use.
2. In parallel, keep a small ExecuTorch WebGPU export prototype as an alternative. Test UNet operators and memory use before committing to it.
3. Use Python/Diffusers only as an offline export/conversion tool or reference implementation; do not require a Python inference server.
4. Keep tokenizer, scheduler, image preprocessing, and orchestration in JavaScript where practical.
5. Load the target .safetensors LoRA locally in the browser and validate its exact keys, tensor shapes, and supported adapter type.
6. Keep model weights out of Git and load them from local files or a configurable Hugging Face source.

Both routes target browser-native inference: model computation occurs inside the browser tab, not on a Python inference server.

## Current implementation

- index.html: separate experiment page with model/runtime research notes and a local LoRA inspection utility.
- lora-inspector.js: reads only the first safetensors header block and reports tensor names, shapes, common LoRA A/B or down/up pairs, possible extended adapter tensors, and embedded training metadata. It does not upload or deserialize tensor payloads.
- No image-generation inference is wired up yet.

## LoRA inspection caveat

The safetensors header can show adapter naming and tensor dimensions, but it does not prove a match to a specific ONNX graph or ExecuTorch export. Text-encoder adapters, UNet adapters, DoRA, LoKr, LoHa, and other variants must be identified and handled explicitly. For standard LoRA pairs, both low-rank matrices must be mapped to the corresponding graph weights. A strength slider must not be exposed until output is verified against a Diffusers reference.

## Validation milestones

- [x] Identify an Illustrious v0.1-derived browser ONNX package with FP16 graph weights-index metadata.
- [ ] Inspect the exact Civitai LoRA version and tensor keys.
- [ ] Verify the Personal Ami LoRA against the package's WAI Illustrious v17 FP16 graph.
- [ ] Prototype ExecuTorch WebGPU export for the SDXL UNet and verify required operator coverage.
- [ ] Build or adapt the ONNX Runtime WebGPU execution pipeline for the split SDXL UNet graphs.
- [ ] Run a baseline prompt in browser WebGPU without a LoRA.
- [ ] Match a reference image from Diffusers within expected precision differences.
- [ ] Apply Personal Ami LoRA at strengths 0.6, 0.75, and 0.9; compare against Diffusers.
- [ ] Investigate quantized UNet execution and adapter application.
- [ ] Add arbitrary dimensions, negative prompt, CFG, sampler/steps, seed, caching, and PNG output.
- [ ] Verify memory use and generation on the target Mac/browser before calling it usable.

## References

- [ExecuTorch WebGPU announcement](https://pytorch.org/blog/executorch-webgpu-pytorch-models-on-the-gpu-in-the-browser/)
- [ExecuTorch WebGPU backend guide](https://docs.pytorch.org/executorch/stable/backends/webgpu/webgpu-overview.html)
- [ExecuTorch WebGPU operator support](https://docs.pytorch.org/executorch/stable/backends/webgpu/webgpu-op-support.html)
- [ONNX Runtime WebGPU guide](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
- [WAI Illustrious ONNX/WebGPU package](https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17)
- [Illustrious XL v0.1 model card](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0)
- [Personal Ami Style - Illustrious](https://civitai.com/models/1309467/personal-ami-style-illustrious)
