# NoobAI XL 1.1 — Browser WebGPU experiment

A new, separate experiment. It does not modify the root FLUX.2 generator, the Illustrious experiment, or the existing Pony Unfiltered Lab.

## Target behavior

The intended user flow is:

1. Open the GitHub Pages app.
2. On the first visit, enter the published browser-model-pack URL and download its ONNX files.
3. The browser stores model graphs and weight shards in its Origin Private File System (OPFS).
4. Generate images locally. Later visits reuse the OPFS copy instead of downloading the multi-gigabyte model again.
5. The app shell/runtime is cached by a service worker; both fast CLIP tokenizers are bundled in the browser pack and loaded directly from its tokenizer JSON/config files.

No Node server, Python inference server, native executable, or local installation is part of the end-user flow. Python is used only by the offline conversion/publishing step.

## Model

The target is **NoobAI XL 1.1** from [Laxhar/noobai-XL-1.1](https://huggingface.co/Laxhar/noobai-XL-1.1). Its model index identifies a `StableDiffusionXLPipeline` with two CLIP text encoders/tokenizers, a U-Net, a VAE and an Euler scheduler. The model card describes training on Danbooru and e621 data.

The model card recommends the quality prefix `masterpiece, best quality, newest, absurdres, highres`, 25–30 steps, CFG 5–6, and Euler a. The app uses those settings as defaults. It deliberately omits the model card's `safe` prefix and omits `nsfw` from the default negative prompt. It also avoids generic negative tags that suppress furry/anthro subjects. Positive and negative prompts are sent as separate SDXL conditioning inputs; clearing the negative prompt disables classifier-free guidance. The app does not apply a prompt filter or output classifier, though the model's own learned behavior cannot be removed by the UI.

The model card's license still applies independently of the application's behavior. It prohibits commercialization, bans uses it calls "unethical or offensive," and requires derivative models and related products to be open-sourced with work details shared. The app does not implement those terms as prompt filters. The export pack preserves the source model card and a terms notice; review both before redistribution or use.

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

This is a one-time conversion step, not an inference server. **The end-user generator runs in the browser; it does not need Python, CUDA, Node.js or a local server.**

The easiest route is the [one-click Google Colab exporter](https://colab.research.google.com/github/DragonFireGames/imagegen/blob/main/noobai-browser-experiment/tools/Export_NoobAI_XL_1.1_to_browser_ONNX.ipynb). Select a CUDA GPU runtime (T4 16 GB VRAM or better), enter a public Hugging Face destination repository, and provide a Hugging Face write token when the notebook asks for it. The notebook installs the dependencies, checks that CUDA is available, converts the model components one at a time, and uploads the browser pack.

The exporter converts both CLIP tokenizers to fast-tokenizer JSON and uploads them with the browser pack. The updated exporter no longer loads the entire Diffusers pipeline into host memory. It loads CLIP-L, CLIP-bigG, the U-Net and VAE sequentially, releasing each before the next. It traces the U-Net at 768² with dynamic spatial axes to keep tracing activations lower, upcasts fragile normalization/softmax calculations, validates the ONNX graphs, and streams external weight data into files no larger than 1.5 GB. A high-RAM Colab runtime is still preferable if conversion exhausts host memory.

For a manual conversion, use a CUDA environment and run:

```bash
git clone https://github.com/DragonFireGames/imagegen.git
cd imagegen
python -m pip install -r noobai-browser-experiment/tools/requirements-export.txt
export HF_TOKEN="YOUR_HUGGING_FACE_WRITE_TOKEN"
python noobai-browser-experiment/tools/export_noobai_onnx.py \
  --device cuda \
  --output noobai-xl-1.1-browser-pack \
  --upload-repo YOUR_ACCOUNT/noobai-xl-1.1-browser-onnx \
  --path-in-repo browser-pack
```

After the upload, open the folder link in this format (the app accepts it directly):

```
https://huggingface.co/YOUR_ACCOUNT/noobai-xl-1.1-browser-onnx/tree/main/browser-pack
```

Enter that folder URL in the app and select **Load and cache model**. The app converts Hugging Face `/tree/` links to `/resolve/` file URLs, then loads `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`, and the matching files in `tokenizer_2/` directly from the pack. It does not ask the original model repository to resolve tokenizer metadata. If you already exported a pack before this tokenizer fix, reopen the Colab notebook and run only the **Repair tokenizer assets on an existing pack (no GPU export)** cell; it patches the existing pack without repeating ONNX conversion.

Do not upload the multi-gigabyte model files to this GitHub Pages repository. GitHub Pages has a published-site size limit; keep the application/runtime here and host the ONNX model pack separately.

## Browser requirements and caching

- Current Chromium with WebGPU.
- JavaScript Promise Integration (JSPI), used by the ONNX Runtime Web JSPI build for Blob-backed external data.
- OPFS support and enough free disk space for the model pack.
- The model components are downloaded by URL into OPFS. Incomplete files are resumed with HTTP Range requests where the host supports them.
- The app requests persistent browser storage, but the browser can deny the request or later evict best-effort storage.
- The page/runtime are cached by the service worker. Tokenizer JSON and config files are fetched directly from the selected pack URL.

The initial pack is a large SDXL model. The U-Net weights alone are about 5.14 GB in FP16. The browser loads each text encoder one at a time, releases both, then loads the U-Net; it releases the U-Net before loading the VAE. This reduces overlapping model-weight allocations. Your 16 GB Mac gives us a more realistic chance than the prior 9 GB setup, but it is not a guarantee: Chrome, GPU allocations, and working activations also consume memory. Start at 768×768. Resolution reduces activations, not the U-Net weights. A quantized U-Net remains a future optimization, not part of this first FP16 pack.

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
9. Verify memory use on the target 16 GB Mac; a successful 768×768 run is not implied by a successful desktop 1024×1024 run.

Until those tests pass, this remains an experimental implementation rather than a verified working generator.
