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
UNet partitions (A, B, C), and the VAE decoder. Model files are cached locally
in the browser; generation and adapter processing happen in the page.

The generator prefers ONNX Runtime Web 1.31.1's JSPI build of the native WebGPU
execution provider when the browser exposes WebAssembly JSPI. The JSPI runtime
is loaded from jsDelivr; the repository's locally bundled Asyncify WebGPU
runtime remains the fallback when JSPI is unsupported or its module import
fails. Add `?runtime=asyncify` to force the local runtime, or
`?runtime=jspi` to require JSPI and surface module-load errors. A JSPI WASM
asset failure after the module imports still appears as a runtime/session error;
use the Asyncify override to diagnose that case.

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
dimensions, step count, CFG scale, seed, and LoRA enablement/strength plus
adapter filename/size metadata. The local model cache and the adapter payload
are not embedded in the JSON.

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

