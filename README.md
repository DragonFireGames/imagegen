# Imagegen

A browser-local FLUX.2 Klein 4B image generator. The application, JavaScript runtimes, WebAssembly files, and tokenizer are served from this repository. The multi-gigabyte ONNX weights are fetched directly from Hugging Face and inference runs on the visitor's WebGPU adapter.

**Live demo:** https://dragonfiregames.github.io/imagegen/

## Features

- FLUX.2 Klein 4B, running locally through ONNX Runtime WebGPU.
- 512×512, 768×768, and 1024×1024 output presets.
- 4, 6, or 8 denoising steps.
- Reproducible seeds and PNG download.
- One short-lived Web Worker per model stage. The encoder, transformer, and decoder workers are terminated after their stage, because releasing an ONNX Runtime WebGPU session alone may not return the allocations to the adapter.
- No generation API, account, API key, telemetry, or app-side prompt-blocking layer.

## Repository layout

- index.html and styles.css: UI.
- src/: tokenizer integration, generation pipeline, and inference worker.
- vendor/onnxruntime-web/: local WebGPU JS runtime and required WASM binaries.
- vendor/transformers/: local Transformers.js distribution.
- models/qwen3-4b-tokenizer/: local tokenizer/configuration files.
- scripts/vendor-assets.sh: reproducible asset-vendoring script.
- .github/workflows/: Pages deployment and dependency-vendoring workflows.

The vendor workflow fetches pinned npm distributions and the upstream tokenizer, commits them into vendor/ and models/, then deploys the static site. To refresh these dependencies, run **Actions → Vendor browser dependencies**.

There are no third-party script or WASM CDNs at application runtime. The source is ordinary browser JavaScript modules; no build step is needed to run it.

## Model weights and GitHub Pages

The ONNX pack is approximately **6.5 GB**: a 2.20 GB int4 text encoder, a 4.14 GB int8 transformer, and a 0.20 GB VAE decoder. These files are intentionally not in this repository. GitHub Pages limits a published site to 1 GB, so including the model weights would make the deployment invalid. See [GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits).

The app downloads only these model files from the licensed [cgb/flux2-klein-4b-onnx-webgpu](https://huggingface.co/cgb/flux2-klein-4b-onnx-webgpu) repository. It uses the publisher's sharded manifests for the large ONNX graphs. Hugging Face is the only external runtime asset host; it is not an inference endpoint.

The model publisher reports cosine similarity of 0.999232 for the int4 text encoder and 0.999426 for the int8 transformer against its PyTorch reference. The 4.14 GB transformer makes adapter memory capacity important. The model stages are run in separate workers to avoid carrying WebGPU allocations from one stage into the next.

## Run locally

Any static HTTP server works. From the repository root:

    python3 -m http.server 8080

Then open http://localhost:8080/ in a WebGPU-capable browser. Do not use file URLs; browsers require a secure context for WebGPU and module workers.

The first generation downloads several gigabytes. Browser caching may retain those files between sessions, but eviction and cache quotas depend on the browser and device.

## Browser requirements

Use a current desktop Chromium-based browser with WebGPU enabled. Safari and Firefox WebGPU support is not assumed by this demo. A GPU adapter with sufficient maximum buffer size and memory is required. If 768×768 fails, try 512×512 first.

## About prompt filtering

The application does not apply its own prompt moderation or blocklist. The current model pack still uses the stock Qwen3 text encoder, which may have learned refusals; lack of an app-side filter is not a guarantee that every prompt will be followed. The separately published uncensored GGUF encoder is not a drop-in replacement for this ONNX graph.

## Licensing

The app source in this repository is MIT-licensed. Third-party runtime distributions and tokenizer assets retain their upstream licenses; see THIRD_PARTY_NOTICES.md and the license files inside vendor/ and models/qwen3-4b-tokenizer/.

The model weights are supplied by cgb under Apache-2.0 as stated in its model card. Read that model repository's license and notices before redistributing the weights separately.
