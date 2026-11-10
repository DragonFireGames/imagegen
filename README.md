# Imagegen

A static, browser-only FLUX.2 Klein 4B demo. Image generation runs locally in the visitor's browser through WebGPU; prompts and generated images are not sent to an inference server.

## Deploy

GitHub Pages is configured through the workflow at .github/workflows/deploy.yml. For the first deployment, open **Settings → Pages → Build and deployment**, set **Source** to **GitHub Actions**, then push to `main` (or run the workflow manually from Actions). Pushes to `main` trigger subsequent deployments. The expected site URL is:

https://dragonfiregames.github.io/imagegen/

## First run

The app downloads model files directly from Hugging Face. The ONNX model pack is about **6.69 GB** in total (text encoder, transformer, VAE decoder and tokenizer). Browser HTTP caching may retain files between sessions, but browser cache quotas and eviction policies vary.

The repository intentionally does not contain model weights. GitHub Pages is used only for the static app; the weights are served by the model repository:

- [cgb/flux2-klein-4b-onnx-webgpu](https://huggingface.co/cgb/flux2-klein-4b-onnx-webgpu)
- [Model card and implementation notes](https://huggingface.co/cgb/flux2-klein-4b-onnx-webgpu/blob/main/README.md)

## Browser and hardware

- Use a current Chromium-based browser with WebGPU enabled.
- The page must be opened over HTTPS or localhost; file URLs are not supported.
- 512×512 is the conservative starting point. 768×768 is supported by the exported model; 1024×1024 is more memory-intensive.
- The model card reports about 22 seconds for 512×512 at 4 steps on one recent discrete GPU. Actual performance varies significantly by GPU, browser, and available memory.
- The text encoder and transformer are run in separate workers so their WebGPU sessions can be released between stages. This is important because WebGPU runtimes may retain allocations after a session is released.

## Model and licensing

The ONNX export is marked Apache-2.0 by its publisher and is based on [Black Forest Labs FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) and Qwen3-4B. See the model card and upstream notices for model-weight terms and attribution. ONNX Runtime Web is MIT-licensed. Transformers.js is used only for tokenization.

## About "uncensored"

The app itself does not add a prompt moderation or prompt-blocking layer. This initial browser pipeline uses the ONNX Qwen3 text encoder included in the model pack. It does **not** yet load the separately published Cordux uncensored GGUF encoder: GGUF is not a drop-in ONNX replacement, and that encoder needs a compatible browser execution/export path before it can be substituted. Do not mistake the absence of an app-side filter for a guarantee that the current encoder has no learned refusals.

## Architecture

- index.html contains the UI, application code, and an inline Web Worker source.
- ONNX Runtime WebGPU is loaded from jsDelivr.
- The Qwen3 tokenizer is loaded with Transformers.js.
- Text encoder, transformer, and VAE weights are fetched from Hugging Face at runtime.
- No server, API key, or hosted image-generation endpoint is used.
