# NoobAI XL 1.1 — Browser WebGPU experiment

A new, separate experiment. It does not modify the root FLUX.2 generator, the Illustrious experiment, or the existing Pony Unfiltered Lab.

## Target behavior

The intended user flow is:

1. Open the GitHub Pages app.
2. On the first visit, enter the published browser-model-pack URL and download its ONNX files.
3. The browser stores model graphs and weight shards in its Origin Private File System (OPFS).
4. Generate images locally. Later visits reuse the OPFS copy instead of downloading the multi-gigabyte model again.
5. The app shell/runtime is cached by a service worker; tokenizer assets are cached by Transformers.js.

No Node server, Python inference server, native executable, or local installation is part of the end-user flow. Python is used only by the offline conversion/publishing step.

## Model

The target is **NoobAI XL 1.1** from [Laxhar/noobai-XL-1.1](https://huggingface.co/Laxhar/noobai-XL-1.1). Its model index identifies a `StableDiffusionXLPipeline` with two CLIP text encoders/tokenizers, a U-Net, a VAE and an Euler scheduler. The model card describes training on Danbooru and e621 data.

The browser app does not prepend a `safe` tag, append `nsfw` to the negative prompt, reject prompts, classify prompt text, or add an output safety checker. Positive and negative prompts are sent as separate SDXL conditioning inputs. The negative prompt can be cleared. The model's own learned behavior is not something the app can remove.

The model card's license still applies independently of the application's behavior. It adds non-commercial and "unethical or offensive content" restrictions. This app does not implement those as prompt filters, but redistribution and use should be checked against the model terms.

## Current implementation status

The app, OPFS model cache, service-worker app-shell cache, ONNX Runtime Web JSPI dependency path, SDXL browser worker, and offline exporter are scaffolded here.

**There is not yet a published NoobAI ONNX browser pack, and the complete export/inference path has not yet been validated against a generated image.** The Generate button is enabled only after a compatible pack is loaded. Do not interpret the presence of the UI or exporter as evidence that a model pack has been successfully converted or tested.

The exporter targets the NoobAI Diffusers components. It emits four ONNX components:
- CLIP-L penultimate hidden states
- CLIP-bigG penultimate hidden states and pooled projection
- SDXL U-Net noise prediction
- VAE decoder

The graph external weights are split into files no larger than 1.5 GB. The browser passes OPFS-backed `File` objects to ONNX Runtime Web's JSPI Blob-backed external-data loader so it does not create one giant JavaScript weight buffer. The text encoders are loaded through WASM one at a time; the U-Net and VAE decoder use WebGPU. The browser sampler applies classifier-free guidance and the checkpoint's EulerDiscreteScheduler update.

## Convert and publish the browser pack

This is an author/developer task, not something ordinary users need to run. Use a CUDA environment with at least 16 GB VRAM and about 24 GB or more host RAM. The exporter streams external ONNX weights when repacking them, but the Diffusers model load and ONNX tracing still have substantial host-memory overhead. A free Colab T4 runtime may not provide enough host RAM; use a high-memory GPU runtime if it OOMs. The exporter deliberately rejects CPU-only conversion.

From a fresh Colab runtime:

```python
!git clone https://github.com/DragonFireGames/imagegen.git
%cd imagegen
!pip install -r noobai-browser-experiment/tools/requirements-export.txt
```

Create a public Hugging Face model repository to host the converted browser pack. Store a write token in the Colab secret/environment as `HF_TOKEN`, then run:

```python
!python noobai-browser-experiment/tools/export_noobai_onnx.py \
  --device cuda \
  --output noobai-xl-1.1-browser-pack \
  --upload-repo YOUR_ACCOUNT/noobai-xl-1.1-browser-onnx \
  --path-in-repo browser-pack
```

The exporter downloads the source NoobAI Diffusers components, exports the graphs, applies fp32 accumulation around normalization/softmax nodes, re-packs ONNX external weights into sub-1.5-GB shards, writes `manifest.json`, and uploads the output folder. Set `HF_TOKEN` in the runtime before using `--upload-repo`.

The resulting model-pack URL for the browser page is:

```
https://huggingface.co/YOUR_ACCOUNT/noobai-xl-1.1-browser-onnx/resolve/main/browser-pack
```

Enter that URL in the app and select **Load and cache model**.

Do not upload the multi-gigabyte model files to this GitHub Pages repository. GitHub Pages has a published-site size limit; keep the application/runtime here and host the ONNX model pack separately.

## Browser requirements and caching

- Current Chromium with WebGPU.
- JavaScript Promise Integration (JSPI), used by the ONNX Runtime Web JSPI build for Blob-backed external data.
- OPFS support and enough free disk space for the model pack.
- The model components are downloaded by URL into OPFS. Incomplete files are resumed with HTTP Range requests where the host supports them.
- The app requests persistent browser storage, but the browser can deny the request or later evict best-effort storage.
- The page/runtime are cached by the service worker. Tokenizer files are small and are cached separately by Transformers.js.

The initial pack is still a large SDXL model. The NoobAI U-Net source weights alone are about 5.14 GB; the FP16 browser export remains approximately that size for the U-Net component. Lowering the resolution to 768×768 reduces activation memory, but does not reduce the U-Net weight allocation. A machine with only about 9 GB total available memory may still fail to allocate the U-Net and browser/runtime overhead. The current exporter is an FP16 proof-of-conversion path, not yet a low-memory model pack. An int4/quantized U-Net is the next optimization to investigate; it must be verified on WebGPU before advertising it as a supported low-memory profile. The available unified/system memory and WebGPU adapter limits ultimately determine whether the current FP16 pack runs.

## Validation checklist

Before calling this finished, the pack must pass all of these on the target browser/hardware:

1. Complete the exporter without ONNX graph or external-data errors.
2. Load the manifest and every graph/shard from the published HTTPS location.
3. Confirm all model files are cached in OPFS and that a second page load performs no large model downloads.
4. Run both text encoders and inspect their output shapes and finite values.
5. Run one U-Net denoising step with finite output.
6. Run a complete 768×768 generation with a fixed seed.
7. Confirm changing the negative prompt changes conditioning, clearing it disables negative conditioning, and changing the seed changes the output.
8. Reload the page, disconnect the network, and generate again from the cached model and app shell.
9. Verify memory use on the target 9 GB-class machine; a successful 768×768 run is not implied by a successful desktop 1024×1024 run.

Until those tests pass, this remains an experimental implementation rather than a verified working generator.
