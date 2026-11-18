# Pixel Foundry — browser-local AI pixel-art generator

This is an AI-only sprite generator. It does not need Python, Node.js, a local API server, an API key, or a hosted inference service.

## How it runs

Open the GitHub Pages site over HTTPS. The browser downloads model assets and caches them with the Cache API. ONNX Runtime Web runs the model on the user's own device:

- WebGPU when available, with a WebAssembly CPU fallback.
- Prompt encoding and sprite generation happen in the browser.
- Prompts and generated images are not sent to an inference server.
- Model files are fetched from Hugging Face; ONNX Runtime Web and the tokenizer module are fetched from jsDelivr.
- Later visits reuse the browser's cached model files when storage is available.

The first model load can take a while. The text encoder is downloaded lazily the first time a prompt is generated. Use Chrome or Edge over HTTPS for the best WebGPU support. CPU fallback works but is significantly slower, especially at 32×32.

## Model

[PixelSprite Canvas v2](https://huggingface.co/ihor-umanets/pixelsprite-canvas-v2) is a 39.1M-parameter autoregressive transformer trained specifically for pixel sprites. It generates transparent 16×16 or 32×32 sprites one pixel at a time, using a 1,024-colour palette. It supports text prompts and can condition colours on a supplied silhouette; this UI currently focuses on text-to-sprite generation.

The browser implementation is adapted from the upstream project's ONNX Runtime Web demo:
https://github.com/umanets/pixelsprite

## Licensing

- Model weights: CC BY-NC 4.0 — non-commercial use only. Do not use these weights or their outputs commercially without resolving the license.
- Upstream inference code: Apache-2.0.
- Text encoder: Xenova/paraphrase-multilingual-MiniLM-L12-v2; see its model repository for its license.
- ONNX Runtime Web: MIT.
- Tokenizer package: see its package license.

This is a different model from Sprite32. Sprite32's current release is a PyTorch pipeline and is not directly browser-ready. PixelSprite was selected because it already has a tested browser ONNX pipeline.
