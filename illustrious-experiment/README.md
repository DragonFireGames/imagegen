# Illustrious browser experiment

This experiment is intentionally isolated from the working FLUX.2 Klein generator.

## Target adapter

[Personal Ami Style - Illustrious](https://civitai.com/models/1309467/personal-ami-style-illustrious) is listed as an Illustrious LoRA trained on Illustrious XL 0.1. Its mirrored listing gives the trigger personal ami and suggests a strength of 0.6–0.9. The mirror identifies its own copy as a reprint, so the original Civitai page remains the source of record.

## Best browser-native lead found

A compatible SDXL-derived ONNX/WebGPU model pack already exists:

- [WAI Illustrious SDXL v17 ONNX package](https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17)
- Its model card declares its base as [OnomaAIResearch/Illustrious-xl-early-release-v0](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0), the Illustrious XL v0.1 base used by the target LoRA.
- The package includes two CLIP text encoders, tokenizer pairs, a three-part SDXL UNet, VAE encoder/decoder, and weights-index.json files that its publisher says are used to apply LoRAs in the browser.
- The full repository is 8.75 GB because it includes both FP16 and INT4 graph variants. The FP16 UNet graphs alone are about 5.13 GB; text encoders are about 1.62 GB and VAE graphs about 0.17 GB.
- The INT4 variants reduce the two largest UNet segments to about 1.82 GB combined, but the model manifest does not declare weights-index files for those INT4 graphs. Therefore they are not the initial LoRA target.

This is a promising runtime starting point, not a claim of tested compatibility. It is WAI Illustrious v17, a fine-tune derived from Illustrious v0.1—not the untouched v0.1 base checkpoint. We still need to apply the Personal Ami adapter and compare output against the same LoRA loaded with Diffusers.

The ONNX pack is under Fair AI Public License 1.0-SD, which contains prohibited-use and source/license notice requirements. Review the pack's [LICENSE.txt](https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/blob/main/LICENSE.txt) before use or redistribution. The original Illustrious v0.1 repository also lists that license and provides a TERM_OF_USE.

## Runtime decision

Do not try to bundle the Python Diffusers runtime into this static site. Diffusers depends on PyTorch and native/runtime facilities that are not directly compilable to browser WASM as-is.

Preferred architecture:

1. Use Python/Diffusers only as an offline export/conversion tool where the pre-exported ONNX pack is insufficient.
2. Run the model's ONNX graphs with ONNX Runtime WebGPU. Tokenization, scheduler, image preprocessing, and orchestration stay in JavaScript.
3. Load the target .safetensors LoRA locally in the browser and validate its exact keys, tensor shapes, and supported adapter type.
4. Use the FP16 graphs first because the package provides weight-index files for those graphs. Investigate INT4 LoRA injection only after FP16 works.
5. Keep model weights out of Git and load them from local files or a configurable Hugging Face source.

This is browser-native inference: model computation occurs inside the browser tab, not on a Python inference server.

## Current implementation

- index.html: separate experiment page with model/runtime research notes and a local LoRA inspection utility.
- lora-inspector.js: reads only the first safetensors header block and reports tensor names, shapes, common LoRA A/B or down/up pairs, possible extended adapter tensors, and embedded training metadata. It does not upload or deserialize tensor payloads.
- No image-generation inference is wired up yet.

## LoRA inspection caveat

The safetensors header can show adapter naming and tensor dimensions, but it does not prove a match to a specific ONNX graph. Text-encoder adapters, UNet adapters, DoRA, LoKr, LoHa, and other variants must be identified and handled explicitly. For standard LoRA pairs, both low-rank matrices must be mapped to the corresponding graph weights. A strength slider must not be exposed until output is verified against a Diffusers reference.

## Validation milestones

- [x] Identify an Illustrious v0.1-derived browser ONNX package with FP16 graph weights-index metadata.
- [ ] Inspect the exact Civitai LoRA version and tensor keys.
- [ ] Verify the Personal Ami LoRA against the package's WAI Illustrious v17 FP16 graph.
- [ ] Build or adapt the ONNX Runtime WebGPU execution pipeline for the split SDXL UNet graphs.
- [ ] Run a baseline prompt in browser WebGPU without a LoRA.
- [ ] Match a reference image from Diffusers within expected precision differences.
- [ ] Apply Personal Ami LoRA at strengths 0.6, 0.75, and 0.9; compare against Diffusers.
- [ ] Investigate quantized UNet execution and adapter application.
- [ ] Add arbitrary dimensions, negative prompt, CFG, sampler/steps, seed, caching, and PNG output.
- [ ] Verify memory use and generation on the target Mac/browser before calling it usable.

## References

- [ONNX Runtime WebGPU guide](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
- [ONNX Runtime Web SDXL documentation](https://huggingface.co/docs/diffusers/optimization/onnx)
- [WAI Illustrious ONNX/WebGPU package](https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17)
- [Illustrious XL v0.1 model card](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0)
- [Personal Ami Style - Illustrious](https://civitai.com/models/1309467/personal-ami-style-illustrious)
