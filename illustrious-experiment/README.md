# Illustrious browser experiment

This experiment is intentionally isolated from the working FLUX.2 Klein generator.

## Pages

- `index.html` is the generation-first interface: model cache setup, local adapter selection, resident engine preparation/release, and image generation.
- `diagnostics.html` contains the separate developer workbench: runtime probes, component/UNet smoke tests, scheduler checks, and LoRA inspection/patch validation. It is intentionally not loaded by the generator page.

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


## Inference performance path

The generator's current WebGPU path prioritizes a repeatable inference loop over experimental graph replay:

- Positive and negative CLIP prompts are concatenated into a batch of two and run in one inference call per encoder when the ONNX graph exposes a compatible batch dimension. The current WAI CLIP graphs report a fixed batch of one, so they use two compatible calls each.
- The three UNet sessions use `graphOptimizationLevel: "all"`. Reusable input tensors are created over the ONNX Runtime-owned WebGPU device; prompt conditioning and time IDs stay in GPU buffers, while only latent and timestep buffers are updated on each denoising step.
- ONNX Runtime is explicitly asked for the browser's `high-performance` WebGPU adapter before the first session is created. During engine preparation, the app runs one ordinary zero-latent UNet pass to compile the model's WebGPU/WGSL pipelines. This warm-up is deliberately **not** graph capture; its temporary A/B outputs are disposed normally. It moves first-run shader compilation into preparation so later image generations can reuse compiled pipelines with the resident sessions.
- UNET_A and UNET_B use GPU-resident outputs for partition handoffs. When the WebGPU device accepts the custom scheduler shader, UNET_C writes into a preallocated GPU output tensor and a WGSL compute pass performs classifier-free guidance, Euler ancestral noise update, and next-step latent scaling entirely on the GPU. The latent state and next UNet input remain GPU-resident across all denoising steps; only the final latent is read back before VAE decoding. If shader compilation or GPU output binding is unsupported, the app logs the reason and falls back to the CPU scheduler.
- CFG exactly `1` uses only the conditional prediction, and CFG exactly `0` uses only the unconditional prediction. Both settings create a batch-one UNet engine instead of evaluating both branches; the engine is keyed by batch size and the selected conditioning branch is uploaded as the single input. Other CFG values retain the standard two-branch formula and batch size two. This optimization is algebraically exact at CFG 0/1, but image appearance naturally differs from guided CFG values.
- The resident UNet cache key includes resolution, adapter identity/strength, and effective batch size. Changing any of these rebuilds the graph-specific sessions.

**Graph capture is currently disabled.** In the first attempt, all three captured sessions completed a zero-latent warm-up, but the next denoising call failed inside ONNX Runtime's WASM/WebGPU bridge with `TypeError: Required member is undefined`. The app disposed the first-run GPU output tensors after warm-up; graph replay depends on the captured buffers remaining valid. Because ORT's WebGPU documentation says output GPU tensors own their buffers and `dispose()` releases them, the old cleanup strategy was not safe to pair with captured replay. A separate ORT WebGPU issue also documents replay problems in a static-shape graph-capture path. To avoid repeating the broken run, engine preparation now creates the sessions without a dummy denoising pass and keeps ordinary WebGPU dispatch plus the reusable GPU inputs and partition handoffs. Revisit capture only with persistent output-buffer ownership and a browser test that exercises repeated replay.

When Personal Ami is enabled, the browser reads its ~218 MiB Safetensors payload into one temporary ArrayBuffer and reuses views into that buffer for the per-layer BF16 factors. This replaces thousands of tiny File.slice/arrayBuffer reads across CLIP-L, CLIP-bigG, and the three UNet patches. The buffer is released from the adapter cache after the resident UNet weights have been prepared; this trades a temporary ~218 MiB RAM allocation for much less file-read overhead. If that allocation fails, the patcher logs the reason and returns to per-tensor reads.

The GPU scheduler uses the ONNX Runtime WebGPU device and a user-owned output tensor. Its FP16 output is unpacked in WGSL, CFG is applied with the same formula as the CPU path, and Euler ancestral coefficients are computed from the same scheduler sigmas. Random noise is still generated on the CPU and uploaded once per step; latent tensors are not read back between steps. The final latent is copied to CPU once for VAE decoding. This path is newly implemented and requires an end-to-end browser test before its speedup or numerical equivalence can be claimed.

**Performance status:** one user benchmark at 768×1024 and 30 steps measured 152.37 seconds summed across A/B/C inference calls, with typical per-step timings around A 3.1s, B 1.6s, and C 0.4s. VAE decode plus canvas conversion took 7.04s. Engine creation also included a 10.62-second zero-latent pass to compile WebGPU pipelines; it is run only when the adapter/resolution/batch engine must be rebuilt, not on every image with a reused engine. These are device-specific timings, not a generalized performance claim. The dominant cost is UNet inference, not prompt parsing or the scheduler.

For per-operator GPU timing, append `?gpuProfile=1` to the generator URL before the page loads. ONNX Runtime Web writes WebGPU profiling records to the browser console with the `[profiling]` prefix. Profiling adds overhead, so do not use that run as a final speed benchmark; use it to identify which kernels dominate, then benchmark again without the query parameter. See [ONNX Runtime Web performance diagnosis](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html#webgpu-profiling).

The app also normalizes the Personal Ami trigger check so an existing `personal ami` tag is not appended a second time. VAE image conversion now reads the FP16 output directly instead of allocating and scanning a second full-size FP32 copy; decoded pixel values and rounding are unchanged.

## Resident inference engine and repeated generation

The generator now supports an explicit **Prepare inference engine** phase:

- It verifies/downloads the full model cache before inference.
- It precomputes the currently selected positive/negative prompt conditioning.
- It creates the three Personal Ami-patched (or unmodified base) UNet WebGPU sessions and keeps them resident in page memory.
- It deliberately does **not** retain the VAE decoder alongside the multi-part UNet. The VAE is loaded for decoding after denoising, its output tensors are disposed, and the session is released so the next generation starts with the UNet as the only resident model.
- It caches up to eight exact prompt-conditioning combinations, keyed by positive prompt, negative prompt, and adapter file identity/strength.
- Repeated generations with the same adapter reuse the resident UNet sessions. Repeated prompts also skip both CLIP encoders. The VAE session is re-created for each decode and released afterward to reduce GPU memory pressure.
- Changing the adapter file or strength causes the UNet sessions to be released and rebuilt with the new fused weights.
- **Release engine** disposes the resident sessions and unlocks the memory-heavy test harness. The downloaded model files remain in Cache Storage.

This is an in-memory browser-tab cache, not a persistent GPU cache: a page reload or browser restart requires the WebGPU sessions to be created again. The model downloads themselves remain in Cache Storage. Keeping the UNet resident reserves several GiB of GPU memory, so release the engine before running the standalone memory-heavy diagnostic tests or using other GPU-intensive applications. The VAE is intentionally transient: it is loaded only for decode and released immediately afterward to avoid competing with the UNet partitions for GPU memory.

The first browser run after this change still needs to validate session retention, adapter switching, explicit release, and repeat-generation timings. No claim is made yet that this lifecycle has been browser-tested.

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

## Single-target browser LoRA patch validation

`lora-clipl-patch-test.js` is the first actual ONNX weight-patching experiment. It selects one shape-compatible BF16 CLIP-L pair from the user-supplied Personal Ami file, reads the pair and its scalar alpha locally, resolves its ONNX initializer through the verified CLIP-L weights index, and applies

`W' = W + strength × (alpha / rank) × (W_up × W_down)`

to the appropriate FP16 weight range. When the index marks the matrix as transposed, the delta is written in the corresponding ONNX storage orientation. The original external-data shard is fetched as a Blob, size-checked and SHA-256-verified against the manifest, then represented as original Blob slices with the patched FP16 byte range inserted. ONNX Runtime receives that patched Blob through its documented `externalData` option.

The test loads the original CLIP-L session and the patched CLIP-L session separately, runs the same deterministic token IDs through both, and compares their `penultimate_hidden_state` outputs. This is deliberately only one matrix: it verifies a real data path before the 980-pair adapter is applied. It does not yet establish full-adapter compatibility, all-pair runtime, or Diffusers equivalence. The actual .safetensors file remains local to the browser.

The current archive snapshot is preserved on branch `archive/pre-personal-ami-lora-2026-11-11`; ongoing LoRA changes are being made on `main`.

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
took 15.9 seconds. This validates CLIP-L only, not the other encoder, UNet,
scheduler, full-model memory use, or LoRA injection.

The VAE decoder test has now passed in the user's browser. It loaded 94.5 MiB
of graph and external weights, accepted an FP16 zero latent with shape
`[1, 4, 128, 128]`, and returned an FP16 `image` tensor with shape
`[1, 3, 1024, 1024]`; all output values were finite. Load plus inference
took 16.1 seconds. The initial FP32 input failure was corrected by using
the graph's actual FP16 input contract.

The CLIP-bigG test has now passed in the user's browser. It loaded 1.29 GiB
of graph and external weights, accepted int64 token IDs with shape
`[1, 77]`, and returned both expected FP16 outputs:
`penultimate_hidden_state [1, 77, 1280]` and `text_embeds [1, 1280]`.
All 99,840 output values were finite. Load plus inference took 79.1 seconds.
Both SDXL text encoders and the VAE decoder have now passed isolated WebGPU
inference tests.

The metadata-only split-UNet inspection has now passed in the user's
browser. All six small files (three ONNX graph protobufs and three weight
indexes) matched the manifest's exact sizes and SHA-256 hashes. It transferred
3.61 MiB and zero tensor shards.

The graph interfaces establish this chain:

- **UNET_A — down path + middle:** inputs `sample` FP16 [B,4,h,w],
  `timestep` FP32 [B], `encoder_hidden_states` FP16 [B,S,2048],
  `text_embeds` FP16 [B,1280], and `time_ids` FP32 [B,6]. It returns
  `hidden`, `emb`, and skip tensors `skip_0` through `skip_8`.
- **UNET_B — first up-path partition:** consumes A's `hidden` and `emb`,
  the conditioning, and `skip_6` through `skip_8`. It returns
  `hidden_out`, which is C's `hidden` input.
- **UNET_C — remaining up path:** consumes B's `hidden_out`, A's `emb`,
  the conditioning, and `skip_0` through `skip_5`. It returns
  `out_sample`, the denoiser prediction.

At a 1024-pixel output size, the expected latent is 128×128; A's observed
intermediate spatial scales are 128, 64, and 32. This gives us a precise
interface for a staged test. The index files contain 383, 319, and 92 entries
for A, B, and C respectively. Their entries map readable Diffusers-style
weight names to ONNX initializer names, shard/offset/length, and both Torch
and ONNX shapes. Some attention projections (including cross-attention
`to_k` and `to_v`) are flagged `transposed: true`. The Personal Ami adapter
must be mapped and its low-rank delta oriented correctly before touching these
weights. The counts are not yet proof that every one of the adapter's 722 UNet
pairs has a direct one-to-one match.

The **UNET_A-only** smoke test has now passed in the user's browser. It loaded
2.32 GiB across three external-data shards, ran a zero-input inference at
1024² in 5.6 seconds (100.5 seconds including load), and checked 26,871,040
values across all eleven FP16 outputs. Every output had the expected shape and
all values were finite. The session was released after the test.

The **UNET_A → UNET_B** staged chain passed in the user's browser. A's
actual FP16 intermediates were copied to CPU memory, A was released, and B
returned finite `hidden_out [1,1280,64,64]`. The latest recorded run retained
8.8 MiB and completed in 187.5 seconds (A 89.7 s, B 97.8 s).

The **full UNET_A → UNET_B → UNET_C** chain also passed. A's required
hidden/embedding/skip outputs were retained as 51.3 MiB of CPU-side FP16 data;
each partition was released before the next loaded. C consumed B's actual
`hidden_out` and A's remaining skips, returning finite FP16
`out_sample [1,4,128,128]`. The latest recorded total was 423.6 seconds
(A 93.1 s, B 230.3 s, C 56.6 s). This proves the full split denoiser forward
chain with zero conditioning, not iterative denoising.

The **VAE encoder** smoke test passed. It loaded the 65.3 MiB component,
accepted a zero-filled FP16 RGB tensor `image [1,3,1024,1024]`, and returned
finite FP16 `latent_mean [1,4,128,128]`. Latent scaling and encode/decode
round-trip equivalence remain to be tested.

Real positive and negative prompt conditioning passed in the user's browser.
Both tokenizer folders produced int64 `[1,77]` sequences (14 attended
positions for the positive prompt and 21 for the negative prompt). CLIP-L
produced finite FP16 `[1,77,768]` hidden states; CLIP-bigG produced finite
FP16 `[1,77,1280]` hidden states and `[1,1280]` pooled embeddings. The test
assembled positive and negative SDXL conditioning pairs:
`encoder_hidden_states [1,77,2048]` and `text_embeds [1,1280]`. Total
tokenization, model loading and inference was 69.8 seconds. No UNet/VAE was
loaded and no image was generated. The tokenizer is built from the pack's
actual `vocab.json`, `merges.txt`, and `special_tokens_map.json` files.

The synthetic Euler ancestral scheduler test passed 28 deterministic steps,
including monotonic sigmas, the sigma-up/down identity, reproducibility, and
finite latent output. It did not run the UNet.

## Runtime decision

Do not try to bundle the Python Diffusers runtime into this static site. Diffusers depends on PyTorch and native/runtime facilities that are not directly compilable to browser WASM as-is.

Preferred sequence:

1. Run the CLIP-L, CLIP-bigG, and VAE decoder component smoke tests in the browser. All three have now passed on the target machine, confirming external-data loading and real WebGPU inference for both text encoders and the decoder.
2. The full staged A → B → C graph chain, VAE encoder, and real positive/negative prompt conditioning have passed isolated tests. Test the integrated scheduler-driven denoising and VAE decode on the target browser, including the all-UNet memory requirement.
3. The no-LoRA WAI v17 SDXL JavaScript pipeline has completed its first full browser generation. All three UNet sessions remained resident, 28 Euler ancestral steps ran with CFG 5, and the VAE produced a 1024×1024 PNG. WAI v17 is a derivative, not the original Illustrious v0.1 training checkpoint.
4. Use Python/Diffusers offline to load the actual Personal Ami adapter, fuse it into the official Illustrious v0.1 base, and generate a reproducible reference image. Do not require a Python inference server.
5. Inspect the WAI FP16 weight-index format and implement a converter for Personal Ami only after all 986 pairs have been mapped and shape-checked. If the derivative graph cannot be validated, export the exact base as a separate browser model instead.
6. Compare browser outputs with Diffusers at strengths 0.6, 0.75, and 0.9 before adding a LoRA-strength control.
7. Keep ExecuTorch as a secondary route until the ONNX baseline is measured; its current WebGPU partitioner can select operators not implemented by the runtime.
8. Keep tokenizer, scheduler, image preprocessing, orchestration, and caching in JavaScript where practical. Keep generated model weights out of Git and load them from local files or a configurable Hugging Face source.

Both routes target browser-native inference: model computation occurs inside the browser tab, not on a Python inference server.

## Current implementation

- index.html: separate experiment page with a prompt/settings panel, progress and cancellation controls, a generated-image preview/PNG download, and detailed performance logging, alongside the research notes and local LoRA inspection utility.
- text-to-image.js: browser-only text-to-image pipeline. It tokenizes and encodes positive/negative prompts through the dual CLIP encoders sequentially, caches conditioning, keeps the split UNet WebGPU sessions and VAE decoder resident between generations, and uses GPU-buffer I/O binding for the A→B→C intermediate handoffs. The final denoiser output is read to CPU for CFG and scheduler math before VAE decoding and PNG rendering. Supports 512–1024 dimensions in multiples of 64, steps, CFG, fixed seed, positive suffix, negative prompt, progress, cancellation between graph calls, and a cumulative A+B+C UNet execution-time counter separate from wall-clock elapsed time. First full baseline generation is verified in the target browser: 1024×1024 PNG, seed 1337, 28 steps, CFG 5. Personal Ami integration is now verified end-to-end in the user's browser at strength 0.75: 66 CLIP-L, 192 CLIP-bigG, and 722 UNet pairs were patched across all five components; all three UNet sessions ran the complete 28-step denoising loop, and the VAE produced a 1024×1024 PNG. The user confirmed the result looked good.
- heavy-model-test-lock.js: page-wide lock that rejects overlapping CLIP, VAE, prompt-conditioning and UNet smoke tests before they download weights or allocate WebGPU sessions.
- lora-inspector.js: reads only the first safetensors header block and reports tensor names, shapes, dtypes, byte-range validity, common LoRA A/B or down/up pairs, rank-dimension mismatches, possible extended adapter tensors, and embedded training metadata. It does not upload or deserialize tensor payloads.
- lora-weight-map-inspector.js: local-header-only matcher for the five FP16 indexes. The corrected browser report matched 980/986 names, with all 980 shape-compatible: 66 CLIP-L, 192 CLIP-bigG, 350 UNET_A, 306 UNET_B, and 66 UNET_C. All 980 targets require ONNX matrix-delta transposition. The six unmatched pairs are exactly CLIP-L layer 11's two MLP and four attention projections; the tested CLIP-L graph exposes `penultimate_hidden_state`, consistent with that final block being absent from the graph index. The mapper now classifies these six separately from unexpected misses. No LoRA tensor payloads were read.
- lora-payload-smoke-test.js: passed on one real local CLIP-L BF16 pair (layer-0 MLP fc1), decoding a rank-32 down/up pair plus alpha and producing a finite, nonzero FP32 delta; it did not patch an ONNX shard or prove model-output equivalence.
- lora-clipl-patch-test.js: single-pair patch test passed in the user's browser. One real BF16 Personal Ami pair was fused into its indexed FP16 CLIP-L initializer, the base shard SHA-256 matched, ONNX Runtime loaded the patched external-data Blob, and identical input IDs produced a measurable output difference.
- lora-full-clipl-test.js: full CLIP-L component patch test passed in the user's browser. All 66 indexed pairs were fused at strength 0.75; six unindexed layer-11 pairs were excluded; the original and patched sessions both ran, with 58,889/59,136 output values changed and output RMS difference 0.0100283.
- lora-payload-integrity-test.js: passed a 4 MiB-chunk scan of all 2,958 tensor records / 114,017,242 BF16 values (228,034,484 payload bytes); all byte ranges and values were valid, with zero non-finite values and no network upload.
- personal-ami-lora.js: shared browser patch engine for all five ONNX components. It maps the adapter's CLIP-L, CLIP-bigG, and Kohya/SGM UNet keys to component indexes; reads local BF16 down/up/alpha tensors; applies scaled FP16 linear or Conv2D deltas with index transpose metadata; verifies base shard size/SHA-256; and returns temporary external-data Blobs. The complete 980-target adapter was successfully applied in the user's browser at strength 0.75 (66/192/350/306/66), with the six unindexed CLIP-L final-block pairs excluded. The current main branch additionally includes an experimental WebGPU compute path for LoRA matrix fusion, with an explicit CPU fallback; shader compilation, numerical agreement, and full-generation speedup still require browser validation. The denoiser also uses experimental ONNX Runtime Web GPU-buffer outputs for split-graph handoffs; cross-session browser validation and speedup measurements remain pending.
- tools/fuse_lora_and_export.py: validates the safetensors header, loads its tensors as a Diffusers state dict, verifies that UNet and both text encoders receive the adapter, then fuses it at the selected strength. The real Personal Ami header report is recorded above; actual full Diffusers fusion still needs to be run.
- tools/README.md: isolated environment and export instructions.
- runtime-probe.js: reports WebGPU capabilities and explicitly flags the built-in ORT LoRA API as WASM-only.
- component-smoke-test.js: CLIP-L graph and dummy-token WebGPU inference; passed in the user's browser (221.3 MiB, 15.9 s, output [1, 77, 768], FP16, finite).
- text-encoder-2-smoke-test.js: passed in the user's browser; 1.29 GiB, int64 [1, 77] input, FP16 outputs [1, 77, 1280] and [1, 1280], all 99,840 values finite, 79.1 seconds.
- vae-smoke-test.js: passed in the user's browser; FP16 input [1, 4, 128, 128], FP16 image output [1, 3, 1024, 1024], all finite, 16.1 seconds.
- unet-metadata-inspector.js: passed in the user's browser; verified all 6 graph/index files against manifest sizes and SHA-256, downloaded 3.61 MiB, fetched 0 UNet tensor shards. Extracted the A → B → C interfaces and weight index layout.
- unet-a-smoke-test.js: passed in the user's browser; 2.32 GiB across 3 shards, inference 5.6 s, total 100.5 s, 26,871,040 FP16 output values across all 11 correctly shaped and finite outputs; session released.
- unet-ab-chain-test.js: passed in the user's browser; A→B actual tensor handoff, 8.8 MiB CPU intermediates, latest recorded total 187.5 s (A 89.7 s, B 97.8 s); B returned finite FP16 hidden_out [1,1280,64,64].
- unet-abc-chain-test.js: passed in the user's browser; sequential full A→B→C chain, one partition resident at a time, 51.3 MiB A intermediates, final finite FP16 out_sample [1,4,128,128], latest recorded total 423.6 s.
- All component/UNet/prompt smoke tests acquire the same shared lock before any download or WebGPU allocation and release it in `finally`; concurrent heavy tests are refused with a visible status error.
- vae-encoder-smoke-test.js: passed in the user's browser; zero RGB input FP16 [1,3,1024,1024] produced finite latent_mean FP16 [1,4,128,128]; latent scaling and round-trip behavior remain unverified.
- prompt-conditioning-smoke-test.js: passed positive and negative prompts through both actual CLIP BPE tokenizers and text encoders. It built finite positive/negative `encoder_hidden_states [1,77,2048]` and pooled `text_embeds [1,1280]`; total 69.8 s. This validates real conditioning tensors, not image generation.
- scheduler-math-smoke-test.js: passed 28 deterministic synthetic Euler ancestral steps; beta/timestep schedule, monotonic sigmas, sigma-up/down identity, repeatability and finite output checked; it did not run the UNet or generate an image.
- unet-resident-memory-preflight.js: high-memory preflight that creates A, B and C ONNX Runtime WebGPU sessions while retaining earlier sessions, validates graph interfaces, then releases them in reverse order. It does not run inference; awaits browser validation.
- Full iterative browser text-to-image inference is wired up and has produced its first successful 1024×1024 PNG. The target browser created all three UNet sessions together (4.79 GiB manifest payload), ran all 28 denoising steps, and completed VAE decoding. The resident engine now retains UNet and VAE sessions between images; GPU-buffer handoffs were added afterward and have not yet been browser-tested.

## Browser text-to-image pipeline

The generator keeps the two CLIP encoders sequential to limit peak memory, then creates a two-item CFG batch (negative first, positive second). At every scheduler step, the same scaled latent is duplicated into batch 2. UNET_A is configured with `preferredOutputLocation: 'gpu-buffer'` for its hidden, embedding, and skip outputs; UNET_B receives the A output Tensor objects directly and keeps `hidden_out` GPU-resident; UNET_C receives B's hidden output plus A's remaining skips directly. The A→B→C intermediate handoffs therefore avoid explicit JS typed-array materialization and re-upload. UNET_C's final `out_sample` remains a CPU output for classifier-free guidance and the Euler ancestral update, so this is a GPU-resident inter-partition handoff, not a fully GPU-resident denoising loop. ORT-owned GPU output tensors are explicitly disposed after their final consumer. GPU tensor location and shapes are checked at runtime; unsupported cross-session binding fails visibly rather than silently falling back. After preparation, all three UNet sessions remain resident between images.

The initial latent and per-step ancestral noise use a deterministic seeded PRNG. The default 1024×1024 / 28-step / CFG 5 settings match the candidate pack's published defaults. The first end-to-end browser run succeeded on the WAI v17 base weights without Personal Ami: 1024×1024, 28 Euler ancestral steps, CFG 5, seed 1337, and a 1.8 MiB PNG. All three UNet sessions were created simultaneously (4.79 GiB manifest payload), remained resident for the denoising loop, and were released before the VAE decoder loaded. CLIP conditioning completed, all 28 steps produced finite latents, and VAE decoding/canvas conversion succeeded. The supplied log reports 9,156.16 seconds total elapsed, while the 28 per-step scheduler intervals sum to about 574.27 seconds; the large gaps between some steps appear to be idle/suspension time rather than model execution. The optional Personal Ami toggle now routes the local adapter through the five-component patcher; integrated browser execution remains unverified.


## Experimental GPU-resident UNet handoffs

The split denoiser now requests selected UNET_A and UNET_B outputs in ONNX Runtime Web's `gpu-buffer` location. The returned ORT Tensor objects are passed directly as inputs to the next partition:

- UNET_A → UNET_B: `hidden`, `emb`, `skip_6`, `skip_7`, `skip_8`
- UNET_A + UNET_B → UNET_C: `emb`, `skip_0`…`skip_5`, `hidden_out`

The app validates each intermediate's FP16 dtype, exact dimensions, GPU location, and exposed `GPUBuffer`. ORT-owned output tensors are disposed after their final downstream use. UNET_C's `out_sample` remains CPU-resident for the current CPU classifier-free guidance and scheduler math.

This relies on ONNX Runtime Web's documented WebGPU I/O binding (`preferredOutputLocation: 'gpu-buffer'` and GPU-backed Tensor inputs). It has passed static syntax and wiring checks, but **still requires browser validation with the actual 1.31.1 bundle and all three large sessions**. If the runtime rejects cross-session GPU tensors, the error is deliberately surfaced; no silent CPU fallback is claimed. The expected benefit is eliminating JS-visible intermediate downloads and re-uploads, but the speedup is not yet measured.

## Offline LoRA fusion and ONNX reference export

The offline helper is documented in [tools/README.md](tools/README.md). It loads the official Illustrious v0.1 Diffusers pipeline, loads a local SDXL `.safetensors` adapter, fuses it at a chosen scale with `safe_fusing=True`, and optionally runs:

```bash
optimum-cli export onnx --model MERGED_DIFFUSERS_DIR --task stable-diffusion-xl --dtype fp16 OUTPUT_DIR
```

The script defaults to strength 0.75 (inside the mirrored Personal Ami recommendation of 0.6–0.9). It keeps generated weights outside Git and writes an `export-metadata.json` describing the base, adapter filename, strength, dtype, versions, and fused components.

This provides a fixed-strength Diffusers reference and a first ONNX export for comparison. **It does not claim that the resulting generic Optimum export is ready for ONNX Runtime WebGPU.** The browser target still needs split graphs, external-data sharding, supported WebGPU operators, and matching numerical output. For an eventual browser release, live LoRA swapping can follow after the fixed-strength model is working.

## LoRA inspection caveat

The safetensors header can show adapter naming and tensor dimensions, but it does not prove a match to a specific ONNX graph or ExecuTorch export. Text-encoder adapters, UNet adapters, DoRA, LoKr, LoHa, and other variants must be identified and handled explicitly. For standard LoRA pairs, both low-rank matrices must be mapped to the corresponding graph weights. The generator exposes the strength control as an experimental option; do not treat its visual strength calibration or numerical equivalence as verified until full-adapter output is compared with a Diffusers reference.

## Validation milestones

- [x] Identify an Illustrious v0.1-derived browser ONNX package with FP16 graph weights-index metadata and inspect its current manifest.
- [x] Run the browser capability probe: the user's browser reports WebGPU, Float16Array, ONNX Runtime Web 1.31.1, and 4 GiB maximum buffer/binding limits.
- [x] Run the 232 MB CLIP-L ONNX WebGPU smoke test: success; output `[1, 77, 768]`, FP16, all finite, 15.9 seconds.
- [x] Run the ~99 MB VAE decoder ONNX WebGPU smoke test: success; FP16 input `[1, 4, 128, 128]`, FP16 output `[1, 3, 1024, 1024]`, all finite, 16.1 seconds.
- [x] Run the ~1.39 GB CLIP-bigG ONNX WebGPU smoke test: success; 1.29 GiB, int64 input [1, 77], FP16 outputs [1, 77, 1280] and [1, 1280], all finite, 79.1 seconds.
- [x] Inspect all three split UNet graphs and weight-index maps: all 6 graph/index files matched manifest SHA-256 and sizes; 3.61 MiB fetched, 0 tensor shards. Interfaces establish A → B → C and expose FP16/FP32 input contracts plus transposed matrix weights.
- [x] Run the single-partition UNET_A WebGPU smoke test at 1024²: passed; 2.32 GiB, 3 shards, 5.6 s inference, 100.5 s total, 26,871,040 FP16 output values all finite and correctly shaped.
- [x] Run the staged UNET_A → UNET_B chain: actual A tensors consumed by B; 8.8 MiB CPU intermediates; A 106.5 s, B 150.5 s, total 275.9 s; finite FP16 hidden_out [1,1280,64,64].
- [x] Run the full sequential UNET_A → UNET_B → UNET_C chain; C returned finite FP16 out_sample [1,4,128,128] after sequential partition loading.
- [x] Run the VAE encoder zero-image WebGPU smoke test; FP16 image [1,3,1024,1024] produced finite latent_mean [1,4,128,128]. Latent scaling and round-trip compatibility remain pending.
- [x] Run positive + negative prompt dual-tokenizer / dual-CLIP conditioning; both FP16 CFG conditioning pairs [1,77,2048] and pooled text embeddings [1,1280] were finite.
- [x] Run header-only Personal Ami → ONNX weights-index mapping; 980/986 pairs matched and were shape-compatible (66 CLIP-L, 192 bigG, 350 UNET_A, 306 UNET_B, 66 UNET_C); six CLIP-L layer-11 pairs are unindexed, consistent with the graph's penultimate output; zero shape mismatches.
- [x] Test whether all three UNet graph sessions can remain resident simultaneously; the target browser created A (2.32 GiB), B (2.25 GiB), and C (224.5 MiB) and retained all three for the complete denoising run (4.79 GiB total manifest payload).
- [x] Run one-pair local BF16 payload test; a real CLIP-L linear pair decoded and produced a finite, nonzero low-rank delta.
- [x] Run full local BF16 payload integrity scan over all 2,958 tensors / 114,017,242 values; all byte ranges and BF16 values passed with zero non-finite values.
- [x] Run the scheduler-only Euler ancestral math test against the Illustrious beta/timestep config; 28 deterministic synthetic steps passed. Integrating the scheduler with iterative real UNet denoising remains pending.
- [x] Add an offline Diffusers fusion / generic ONNX reference export script (Python syntax and `--help` pass; synthetic BF16 Safetensors test reproduced the real scan's 2,958 keys, 986 rank-32 pairs, 986 alpha keys, and three component counts; deliberately malformed rank was rejected). Real Personal Ami loading/fusion is still pending.
- [x] Inspect the exact Personal Ami LoRA header and tensor-key layout in the browser (actual report recorded above).
- [ ] Load and fuse the actual Personal Ami tensors in Diffusers against the official Illustrious v0.1 base.
- [ ] Verify the Personal Ami LoRA against the package's WAI Illustrious v17 FP16 graph (this derivative is not the exact v0.1 training checkpoint).
- [ ] Prototype ExecuTorch WebGPU export for the SDXL UNet and verify required operator coverage.
- [x] Complete the staged A → B → C WebGPU denoiser graph chain with zero conditioning.
- [x] Validate real positive/negative prompt conditioning and construct both finite SDXL CFG conditioning pairs.
- [ ] Validate VAE encode/decode scaling and round-trip compatibility independently; successful image generation confirms the chosen decode scale can produce a visually coherent image but is not a formal encoder/decoder round-trip test.
- [x] Run a baseline prompt in browser WebGPU without a LoRA; first 1024×1024, 28-step, CFG 5, seed 1337 image generated successfully and saved as a 1.8 MiB PNG.
- [ ] Match a reference image from Diffusers within expected precision differences.
- [x] Run the single-pair CLIP-L ONNX shard patch test; the patched Blob loaded and changed the graph output.
- [x] Run the full 66-pair CLIP-L ONNX patch test; all indexed targets were applied and the patched CLIP-L session produced a measurable output change.
- [x] Add an optional Personal Ami adapter toggle and strength control to the generator; the shared patcher is wired for all 980 indexed targets (66 CLIP-L, 192 bigG, 722 split across UNet A/B/C), with browser end-to-end validation pending.
- [x] Run the integrated 980-target Personal Ami adapter in the browser at strength 0.75; all five components patched and a complete 28-step 1024×1024 generation succeeded.
- [ ] Compare full-adapter browser outputs at strengths 0.6, 0.75, and 0.9 against a Diffusers reference.
- [ ] Validate the experimental WebGPU LoRA fusion shader against the CPU patcher on representative linear/Conv2D targets, then compare full component patch times.
- [ ] Investigate quantized UNet execution and adapter application.
- [x] Add dimension controls (512–1024 multiples of 64), negative prompt, CFG, Euler ancestral steps, seed, progress, cancellation, performance logs, and PNG output.
- [x] Verify end-to-end generation on the target Mac/browser with Personal Ami: all five adapter components patched, all three UNet sessions resident, 28 denoising steps, VAE decode, and PNG output succeeded. Optimization of adapter fusion and denoising throughput remains open.

## References

- [ExecuTorch WebGPU announcement](https://pytorch.org/blog/executorch-webgpu-pytorch-models-on-the-gpu-in-the-browser/)
- [ExecuTorch WebGPU backend guide](https://docs.pytorch.org/executorch/stable/backends/webgpu/webgpu-overview.html)
- [ExecuTorch WebGPU operator support](https://docs.pytorch.org/executorch/stable/backends/webgpu/webgpu-op-support.html)
- [ONNX Runtime WebGPU guide](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
- [WAI Illustrious ONNX/WebGPU package](https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17)
- [Illustrious v0.1 scheduler configuration](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/blob/b41f1c324e32e03f9e104bd9ace0992e4e135597/scheduler/scheduler_config.json)
- [Diffusers Euler ancestral scheduler implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_ancestral_discrete.py)
- [Illustrious XL v0.1 model card](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0)
- [Personal Ami Style - Illustrious](https://civitai.com/models/1309467/personal-ami-style-illustrious)

## Persistent model cache and portable bundles

- `model-cache.js` prepares generation assets before inference: CLIP-L, CLIP-bigG, UNET_A/B/C, VAE decoder, ONNX graphs, weight indexes, and tokenizer files. It requests persistent browser storage where supported, estimates required quota, downloads with bounded concurrency, verifies file sizes, and stores responses in Cache Storage. Existing cached files are reused on subsequent generations.
- The generator runs the same preparation step automatically at the beginning, so downloads no longer interrupt prompt encoding or denoising. The separate **Download & cache all model files** control lets users front-load this work before generating.
- **Export portable model bundle** creates one `.wkimodel` archive (a versioned binary container, not ZIP) containing all required model files. **Import model bundle** validates bundle metadata and file sizes, checks SHA-256 for files up to 256 MiB, then imports assets into Cache Storage. Large shard hashes remain verified by the existing LoRA patch path when those shards are patched; bundle import avoids materializing multi-gigabyte files into one ArrayBuffer.
- The Personal Ami safetensors file is deliberately not included in the bundle. It remains a local user-provided file.
- Browser quota, persistence grants, Cache Storage limits, and available disk space vary. Bundle export may require several GiB of free space in addition to the cached model.

## Latest browser follow-up

- Fixed the `ReferenceError: log is not defined` in the LoRA shard patcher by passing the generation logger through to `patchShard`.
- Restored the dedicated Conv2D patch implementation for convolutional LoRA targets; the experimental WebGPU matrix path is used only for 2-D linear weights.
- Added experimental WebGPU linear fusion with CPU fallback. Its numerical equivalence and speedup remain unverified in the browser.
- The model cache and portable bundle code passes static JavaScript syntax checks. Actual multi-gigabyte download, Cache Storage quota behavior, bundle round-trip, and repeat-generation cache hits still require browser testing.