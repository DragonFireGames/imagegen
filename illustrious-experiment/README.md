# Illustrious browser experiment

This experiment is intentionally isolated from the working FLUX.2 Klein
generator in the rest of this repository.

## Pages

- `index.html` is the text-to-image interface: model cache setup, prompt and
  sampling controls, optional local LoRA selection, engine preparation/release,
  image generation, and JSON settings import/export.
- `infill.html` is a separate brush-mask image editing experiment. Upload an
  image, paint/erase the region to replace, set an edit prompt and strength, and
  run the local Illustrious model.
- `diagnostics.html` contains standalone browser checks for the model runtime,
  Safetensors headers and payloads, SDXL LoRA-to-ONNX weight mapping, and
  single/full CLIP-L patch validation.

## Model and runtime

The browser app runs WAI Illustrious v17 as a split SDXL ONNX graph using
ONNX Runtime Web and WebGPU. It loads dual CLIP text encoders, three sequential
UNet partitions (A, B, C), and the VAE decoder. Model files are stored in Cache Storage; generation and adapter processing
happen in the page. The explicit **Download & cache model** action downloads
and verifies the complete model set, including both CLIP encoders, all three
UNet partitions, the VAE encoder and decoder, tokenizer files, weight indexes,
external-data shards, and the pinned scheduler configuration. New portable
`.wkimodel` bundles include the scheduler and VAE encoder as well.

The generator prefers the locally vendored ONNX Runtime Web 1.31.1 JSPI build
of the native WebGPU execution provider when the browser exposes WebAssembly
JSPI. The repository's locally bundled Asyncify WebGPU runtime remains the
fallback when JSPI is unsupported or its module import fails. Both runtime
variants and their WASM assets are cached by the experiment's service worker;
the generator does not import ONNX Runtime from a CDN. Add
`?runtime=asyncify` to force the local Asyncify runtime.

The model cache is separate from inference: generation reads ONNX graph bytes
and external weight shards from Cache Storage and passes local buffers/Blob
objects to ONNX Runtime rather than giving it Hugging Face URLs. Tokenizer
resources and the scheduler are read through the same local-first cache
helper. The generator no longer runs a full model-cache verification/download
pass for every image. If a resource is missing or fails its cached-size check,
the missing path is logged and a fallback download is made once and cached;
after a successful **Download & cache model** setup, ordinary generation
should not need model-host HTTP requests.

The experiment also registers a service worker that precaches the page modules,
Transformers tokenizer runtime, ONNX Runtime modules, and corresponding WASM
assets. This keeps the app shell and inference runtime available offline after
the initial app-shell setup. Model data itself remains in the separate model
Cache Storage cache.

The three UNet partitions pass intermediate tensors through GPU-resident
outputs where supported by the current runtime. The Euler ancestral scheduler
keeps classifier-free guidance and latent updates on WebGPU when its GPU path
is available. A resident inference engine can be prepared once and reused for
multiple generations. The VAE decoder is loaded for image decoding and released
afterward to reduce competition with the resident UNet sessions.

Use `?gpuProfile=1` to enable ONNX Runtime WebGPU operator profiling. Profiling
adds overhead, so use it to identify expensive operators, not to record final
generation benchmarks.

Use `?graphCapture=1` to try WebGPU graph capture for the three fixed-shape
UNet sessions. The first disposable warm-up dispatch records the graphs. If
session creation or warm-up fails, the generator releases the attempted
sessions and retries with ordinary WebGPU dispatch. Graph capture may reduce
per-run CPU command-submission overhead; it does not reduce the UNet's GPU
computation. It is opt-in because not every model/operator combination is
capture-compatible.

Use `?layout=NHWC` to request NHWC as the preferred layout for the three UNet
sessions. The default remains NCHW. Layout preference can change which kernels
or layout conversions the WebGPU provider selects, so benchmark it on the
target GPU; NHWC is not assumed to be faster. It does not change model weights
or tensor precision. This option can be combined with graph capture.

Use `?keepVae=1` to retain the VAE decoder after the first generated image.
This avoids recreating its session for subsequent generations, at the cost of
keeping the decoder's GPU allocations resident alongside the UNet. This is
opt-in; the default releases VAE after each image. Releasing the inference
engine frees it, and changing the adapter or resolution releases it before
rebuilding the UNet.

## Multi-image GPU batches

Set **Batch** to generate 1–20 output images. **Images per GPU pass** controls
how many independent latents run through the split UNet together (1–4). The
default is 1, which preserves sequential inference. Start with 2 and increase
only if the GPU has enough memory; UNet activation and persistent output-buffer
memory scale with the model batch. Brush infill stays at one image per pass.

Images in a GPU pass share the prompt, negative prompt, dimensions, sampler
settings, adapter, and cached CLIP conditioning. Each image retains its own
initial noise, per-step ancestral noise, latent state, and seed. Classifier-free
guidance is still computed independently for every image. The split UNet
partitions run once per denoising step for the whole image group, after which
each image is decoded separately. The final partial group is padded with a
duplicate latent to keep the resident graph shape; padded output is discarded.

Each result is added to the gallery as soon as it is decoded, with its own PNG
download link and actual seed. The resident UNet sessions and conditioning
cache are reused across groups with the same shape. Changing the GPU batch
size requires a different UNet session shape and therefore rebuilds the
resident sessions.

For a fixed starting seed, each requested image uses the next 32-bit seed in
sequence (starting seed, starting seed + 1, and so on). Padding does not consume
an additional output seed. The batch is rejected if the requested seed sequence
would exceed the maximum 32-bit seed. With seed `-1`, each requested image
gets a separately generated random seed. Cancelling stops the batch and keeps
all images that already completed; an error also stops further images without
removing earlier results.

## Brush infill experiment

The standalone brush page uses the same WAI Illustrious v17 ONNX model pack.
It encodes the source image through the pack's VAE encoder, applies a soft brush
mask at latent resolution, and blends the source latents back into unpainted
areas during denoising. The final composite also restores unpainted source
pixels exactly. The VAE encoder is included in the browser cache only when an
infill generation is requested; the regular text-to-image path is unchanged.

This is a masked-latent experiment using the regular four-channel SDXL UNet,
not a purpose-trained nine-channel inpainting checkpoint. It is useful for
testing the brush workflow and local infill behavior, but may produce weaker
semantic continuity around the painted boundary than a dedicated inpainting
model. Infill uses the CPU latent scheduler path so the source latents can be
blended at each denoising step; the normal generator retains its GPU scheduler.

## Local SDXL LoRA adapters

The generator accepts a user-selected local `.safetensors` adapter. The file
stays in the browser and is not uploaded. Enable **Apply LoRA**, choose a
strength from 0 to 2, and select the file. Trigger words are never inserted
automatically: enter any trigger required by the adapter in the prompt yourself.

The browser patcher maps compatible Kohya-style SDXL LoRA tensor names to this
model pack's ONNX weight indexes. It can apply adapters to the supported subset
of CLIP-L, CLIP-bigG, and the split UNet partitions. It supports paired down/up
or A/B factors in FP16, BF16, and FP32, using per-layer alpha values where
present and the adapter's `ss_network_alpha` metadata (or rank) as the fallback.
The patcher checks tensor shapes, storage orientation, indexed byte ranges, and
base shard integrity before creating patched external-data blobs.

This does **not** mean every `.safetensors` file is an interchangeable SDXL
LoRA. The adapter must use a supported SDXL architecture, tensor naming/layout,
and target weights represented in the WAI v17 indexes. Other architectures,
DoRA/LoHa/LoKr/OFT, unsupported convolution layouts, and incompatible target
shapes are not applied by this patcher.

The settings JSON records the adapter filename and byte size, not its contents
or a cryptographic digest. After importing settings, reselect the same local
file. The adapter's own licensing and usage conditions remain the responsibility
of the user supplying it.

## Settings JSON

The generator can export and import prompt, negative prompt, positive suffix,
dimensions, step count, CFG scale, seed, batch count, and LoRA enablement/strength
plus adapter filename/size metadata. Version 4 settings files store the image count per GPU pass and retain backward
compatibility with versions 2 and 3; older files default to one image per pass.
The local model cache and the adapter payload are not embedded in the JSON.

## Offline reference exporter

`tools/fuse_lora_and_export.py` is a separate build-time Diffusers utility.
It loads a local SDXL LoRA state dictionary, checks which supported SDXL
components are present, and fuses the adapter into the configured Illustrious
base model. It can optionally export a reference ONNX package through Optimum.
This export is not automatically compatible with the browser's split WebGPU
manifest and weights indexes.

See [tools/README.md](./tools/README.md) for setup, usage, and limitations.

## Important constraints

- Keep generated model weights and fused multi-gigabyte exports outside Git.
- The browser adapter patcher modifies model weights in memory only; it does
  not alter the hosted model files or the user's local adapter file.
- A change in adapter, strength, resolution, or CFG batch mode can require
  rebuilding the resident UNet sessions.
- The reported WebGPU buffer limits are per-buffer limits, not measurements of
  total available GPU memory.
- Browser performance and memory behavior depend on the browser, GPU, driver,
  and operating system. Use repeatable settings and compare warm runs separately
  from first-load, model-patching, and session-creation time.

