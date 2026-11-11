# Illustrious browser experiment

This experiment is intentionally isolated from the working FLUX.2 Klein generator.

## Target adapter

[Personal Ami Style - Illustrious](https://civitai.com/models/1309467/personal-ami-style-illustrious) is listed as an Illustrious LoRA trained on Illustrious XL 0.1. Its mirrored listing gives the trigger personal ami and suggests a strength of 0.6–0.9. The mirror identifies its own copy as a reprint, so the original Civitai page remains the source of record.

## Actual Personal Ami adapter header

The user ran the browser inspector on `personal_ami_style_illustrious-000006.safetensors` and supplied this real report:

- File size: 217.9 MB
- 2,958 tensors, 114,017,242 parameters
- 986 complete LoRA down/up pairs; zero incomplete pairs and zero rank-shape mismatches
- Component tensor counts: text encoder 1 = 216, text encoder 2 = 576, UNet = 2,166
- Embedded metadata: `ss_sd_model_name=889818.safetensors`, `ss_base_model_version=sdxl_base_v1-0`, `ss_network_module=networks.lora`, `ss_network_dim=32`, `ss_network_alpha=16`, `ss_v2=False`, `modelspec.architecture=stable-diffusion-xl-v1-base/lora`, `modelspec.title=personal_ami_style_illustrious`
- Sample keys use Kohya-style `lora_te1_`, `lora_te2_`, and `lora_unet_` prefixes with BF16 down/up tensors.

The component counts divide evenly into 72 text_encoder pairs, 192 text_encoder_2 pairs, and 722 UNet pairs (986 total), with one scalar alpha entry per pair implied by the 2,958-tensor total. The header reports rank 32 and alpha 16, so the Kohya alpha/rank ratio is 0.5; Diffusers needs those per-layer `.alpha` values preserved as network alphas. The author listing mirrored by TensorHub explicitly says the LoRA was trained on Illustrious 0.1 and recommends `personal ami` with strength 0.6–0.9. The embedded `sdxl_base_v1-0` value is the generic SDXL base-version tag; it does not by itself identify the specific fine-tuned checkpoint.

**Compatibility assessment:** the real file has the expected SDXL architecture and all three adapter groups, in the Kohya layout Diffusers knows how to convert. This is strong evidence that it is loadable as an SDXL LoRA. The remaining test is an actual Diffusers load/fuse against the official Illustrious v0.1 pipeline, followed by output comparison. The user has not uploaded the raw model file into this repository or chat runtime; the report above is the user's browser-side scan.

## Browser-native runtime candidates

### Option A: pre-exported ONNX + ONNX Runtime WebGPU

A compatible SDXL-derived ONNX/WebGPU model pack already exists:

- [WAI Illustrious SDXL v17 ONNX package](https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17)
- Its model card declares its base as [OnomaAIResearch/Illustrious-xl-early-release-v0](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0), the Illustrious XL v0.1 base used by the target LoRA.
- The package includes two CLIP text encoders, two tokenizers, a three-part SDXL UNet, VAE encoder/decoder, and weights-index.json files that its publisher says are used to apply LoRAs in the browser.
- The full repository is 8.75 GB because it includes both FP16 and INT4 graph variants. The FP16 UNet graphs alone are about 5.13 GB; text encoders are about 1.62 GB and VAE graphs about 0.17 GB.
- The INT4 variants reduce the two largest UNet segments to about 1.82 GB combined, but the model manifest does not declare weights-index files for those INT4 graphs. Therefore they are not the initial LoRA target.

This is a promising runtime and graph-layout reference, not the final target checkpoint and not a claim of tested compatibility. It is WAI Illustrious v17, a fine-tune derived from Illustrious v0.1—not the untouched v0.1 checkpoint. The Personal Ami adapter is reported as trained on Illustrious v0.1, so its exact result must be compared on the official v0.1 base before accepting the WAI v17 derivative as a compatible substitute.

The ONNX pack and the original Illustrious v0.1 base are under Fair AI Public License 1.0-SD. The license issuer explicitly says this is **not a free-software license** because of its Prohibited Uses section. It prohibits specified illegal or harmful uses (including exploiting minors, targeted harassment, and some forms of discriminatory or harmful decision-making), and requires license/source notices to accompany copies. If a modified model is exposed over a network, the license requires a reasonable way for users to obtain the corresponding source; if users interact only with a derived model, a source offer for that derived model may suffice. Read the [license text](https://freedevproject.org/faipl-1.0-sd/), the ONNX pack's [LICENSE.txt](https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/blob/main/LICENSE.txt), [NOTICE.txt](https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/blob/main/NOTICE.txt), and the base model's [terms](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/blob/main/TERM_OF_USE) before redistribution or network serving. This experiment links to upstream weights and does not copy them into this repository. The Personal Ami LoRA is a separate work; its author's license/permission to redistribute has not been established, so keep it user-supplied and local.

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

### Browser LoRA format reference

The same Hugging Face publisher also exposes a browser-ready [SDXL-Lightning LoRA package](https://huggingface.co/aaaaaeeeee/blip-lora-sdxl-lightning8). It repacks an ordinary Kohya-format `.safetensors` adapter into `lora.json` plus `lora.bin` (FP16 factors; the published example reports rank 64, alpha 8, and 788 UNet layers). This demonstrates the packaging pattern for browser use, but it does not supply a generic ONNX Runtime LoRA API and it is not the Personal Ami adapter.

For Personal Ami, the conversion step must map its exact source tensor names to the target Illustrious graph weights. Text-encoder factors, UNet factors, fused projections, convolutional weights, rank/alpha, and any unsupported adapter variants all need explicit handling. Do not use the Lightning adapter's layer table for another LoRA.

## ONNX Runtime's built-in LoRA API (not usable for this WebGPU path)

The repository currently vendors ONNX Runtime Web **1.31.1**, and its JavaScript
bundle exports `LoraAdapter`. That export is misleading for this use case:
the official API documentation explicitly limits it to the **WebAssembly
execution provider**. It cannot be activated on the WebGPU sessions used by
this experiment. It also expects ONNX Runtime's own `.onnx_adapter` format,
not the target Kohya `.safetensors` file.

Therefore, do not use the built-in `LoraAdapter` API for the browser GPU
pipeline. The earlier browser probe only reported that the class was present
in the JavaScript bundle; the probe now labels it as WASM-only.

Reference: [ONNX Runtime LoraAdapter API](https://onnxruntime.ai/docs/api/js/interfaces/LoraAdapter.html).

## Candidate package manifest and browser limits

The candidate package's actual `manifest.json` lists these FP16 components:

| Component | External weight bytes (approx.) |
|---|---:|
| CLIP-L text encoder | 232 MB |
| CLIP-bigG text encoder | 1.39 GB |
| UNet A | 2.49 GB |
| UNet B | 2.41 GB |
| UNet C | 235 MB |
| VAE decoder | 99 MB |
| VAE encoder | 68 MB |

The two additional INT4 UNet graphs total about 1.82 GB. The full repository
size is 8.75 GB because it contains both FP16 and INT4 variants. The FP16
component weights total approximately 6.93 GB. External-data shards are each
below 1,000,000,000 bytes. The FP16 text encoders and the three FP16 UNet
graphs have `weights-index.json` sidecars; the INT4 UNet entries do not.

The user's browser-side probe reports:

- ONNX Runtime Web 1.31.1
- `Float16Array` available
- WebGPU adapter available
- `maxBufferSize`: 4.00 GiB
- `maxStorageBufferBindingSize`: 4.00 GiB
- `maxComputeWorkgroupsPerDimension`: 65,535

The 4 GiB values are per-buffer/binding limits, **not** a statement that 4 GiB
of total GPU memory is available. A session can allocate multiple buffers, and
the full pipeline's combined CPU/GPU memory use remains unknown.

## Browser-side LoRA format

The WAI package's `weights-index.json` files are a publisher-specific sidecar
format that maps model weights into external-data shards. They are distinct
from ONNX Runtime's `.onnx_adapter` format. The model card says this sidecar
format is used by its own browser LoRA implementation, but that does not prove
that Personal Ami's Kohya tensor names and shapes map to the WAI graphs.

The conversion still needs to:

1. Map each Personal Ami `lora_te1_`, `lora_te2_`, and `lora_unet_` pair to
   the correct graph initializer.
2. Check orientation, fused projections, convolution kernels, and the
   per-layer alpha/rank scaling.
3. Convert BF16 LoRA tensors to a browser-supported representation without
   changing their values beyond expected FP16 rounding.
4. Compare the converted result against a Diffusers reference image.

Do not use the SDXL-Lightning adapter's layer map for Personal Ami. Do not
expose a LoRA strength slider until the converted adapter has passed a numerical
comparison against the same base model and strength in Diffusers.

## First browser model test

The page now includes a **CLIP-L ONNX WebGPU smoke test** in
`component-smoke-test.js`. It reads the candidate manifest, loads only the
232 MB CLIP-L component and its external-data shard, creates a WebGPU session,
and runs one dummy-token inference. It does not download the bigG encoder,
UNet, VAE, or Personal Ami LoRA, and it does not create an image.

The user has now run this test successfully in Chrome. It loaded the CLIP-L
graph and its 221.3 MiB component download, created a WebGPU session, and ran
one dummy-token inference. The output was `penultimate_hidden_state`, shape
`[1, 77, 768]`, dtype `float16`, with all finite values; load plus inference
took 15.6 seconds. This validates CLIP-L only, not the other encoder, UNet,
scheduler, full-model memory use, or LoRA injection.

A second independent test is now available for the VAE decoder. It loads about
99 MB of graph and weights and runs a zero-latent decode. It has not yet been
run in the user's browser.

## Runtime decision

Do not try to bundle the Python Diffusers runtime into this static site. Diffusers depends on PyTorch and native/runtime facilities that are not directly compilable to browser WASM as-is.

Preferred sequence:

1. Run the 232 MB CLIP-L component smoke test in the browser. Confirm that the manifest's external-data mapping and the WebGPU execution provider work on the actual target machine.
2. Inspect the I/O metadata of all split UNet graphs and test the CLIP-bigG and VAE components individually. Do not download every component until the preceding stage works.
3. Build the no-LoRA WAI v17 SDXL pipeline in JavaScript and generate a baseline image. Keep in mind that WAI v17 is a derivative, not the original Illustrious v0.1 training checkpoint.
4. Use Python/Diffusers offline to load the actual Personal Ami adapter, fuse it into the official Illustrious v0.1 base, and generate a reproducible reference image. Do not require a Python inference server.
5. Inspect the WAI FP16 weight-index format and implement a converter for Personal Ami only after all 986 pairs have been mapped and shape-checked. If the derivative graph cannot be validated, export the exact base as a separate browser model instead.
6. Compare browser outputs with Diffusers at strengths 0.6, 0.75, and 0.9 before adding a LoRA-strength control.
7. Keep ExecuTorch as a secondary route until the ONNX baseline is measured; its current WebGPU partitioner can select operators not implemented by the runtime.
8. Keep tokenizer, scheduler, image preprocessing, orchestration, and caching in JavaScript where practical. Keep generated model weights out of Git and load them from local files or a configurable Hugging Face source.

Both routes target browser-native inference: model computation occurs inside the browser tab, not on a Python inference server.

## Current implementation

- index.html: separate experiment page with model/runtime research notes and a local LoRA inspection utility.
- lora-inspector.js: reads only the first safetensors header block and reports tensor names, shapes, dtypes, byte-range validity, common LoRA A/B or down/up pairs, rank-dimension mismatches, possible extended adapter tensors, and embedded training metadata. It does not upload or deserialize tensor payloads.
- tools/fuse_lora_and_export.py: validates the safetensors header, loads its tensors as a Diffusers state dict, verifies that UNet and both text encoders receive the adapter, then fuses it at the selected strength. The real Personal Ami header report is recorded above; actual full Diffusers fusion still needs to be run.
- tools/fuse_lora_and_export.py: offline Diffusers LoRA fusion with optional Optimum SDXL ONNX export. It has not been run against the real Personal Ami weights.
- tools/README.md: isolated environment and export instructions.
- runtime-probe.js: reports WebGPU capabilities and explicitly flags the built-in ORT LoRA API as WASM-only.
- component-smoke-test.js: CLIP-L graph and dummy-token WebGPU inference; passed in the user's browser (221.3 MiB, 15.6 s, output [1, 77, 768], FP16, finite).
- vae-smoke-test.js: loads only the VAE decoder and runs a zero-latent WebGPU inference; awaits browser validation.
- Full image-generation inference is not wired up yet.

## Offline LoRA fusion and ONNX reference export

The offline helper is documented in [tools/README.md](tools/README.md). It loads the official Illustrious v0.1 Diffusers pipeline, loads a local SDXL `.safetensors` adapter, fuses it at a chosen scale with `safe_fusing=True`, and optionally runs:

```bash
optimum-cli export onnx --model MERGED_DIFFUSERS_DIR --task stable-diffusion-xl --dtype fp16 OUTPUT_DIR
```

The script defaults to strength 0.75 (inside the mirrored Personal Ami recommendation of 0.6–0.9). It keeps generated weights outside Git and writes an `export-metadata.json` describing the base, adapter filename, strength, dtype, versions, and fused components.

This provides a fixed-strength Diffusers reference and a first ONNX export for comparison. **It does not claim that the resulting generic Optimum export is ready for ONNX Runtime WebGPU.** The browser target still needs split graphs, external-data sharding, supported WebGPU operators, and matching numerical output. For an eventual browser release, live LoRA swapping can follow after the fixed-strength model is working.

## LoRA inspection caveat

The safetensors header can show adapter naming and tensor dimensions, but it does not prove a match to a specific ONNX graph or ExecuTorch export. Text-encoder adapters, UNet adapters, DoRA, LoKr, LoHa, and other variants must be identified and handled explicitly. For standard LoRA pairs, both low-rank matrices must be mapped to the corresponding graph weights. A strength slider must not be exposed until output is verified against a Diffusers reference.

## Validation milestones

- [x] Identify an Illustrious v0.1-derived browser ONNX package with FP16 graph weights-index metadata and inspect its current manifest.
- [x] Run the browser capability probe: the user's browser reports WebGPU, Float16Array, ONNX Runtime Web 1.31.1, and 4 GiB maximum buffer/binding limits.
- [x] Run the 232 MB CLIP-L ONNX WebGPU smoke test: success; output `[1, 77, 768]`, FP16, all finite, 15.6 seconds.
- [ ] Run the ~99 MB VAE decoder ONNX WebGPU smoke test.
- [ ] Inspect CLIP-bigG and all three split UNet graphs before loading them.
- [x] Add an offline Diffusers fusion / generic ONNX reference export script (Python syntax and `--help` pass; synthetic BF16 Safetensors test reproduced the real scan's 2,958 keys, 986 rank-32 pairs, 986 alpha keys, and three component counts; deliberately malformed rank was rejected). Real Personal Ami loading/fusion is still pending.
- [x] Inspect the exact Personal Ami LoRA header and tensor-key layout in the browser (actual report recorded above).
- [ ] Load and fuse the actual Personal Ami tensors in Diffusers against the official Illustrious v0.1 base.
- [ ] Verify the Personal Ami LoRA against the package's WAI Illustrious v17 FP16 graph (this derivative is not the exact v0.1 training checkpoint).
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
