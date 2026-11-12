# Background Removal Lab

A separate browser-local experiment for removing image backgrounds. It supports AI segmentation, a non-AI connected color-key tool, and manual mask cleanup.

## Run

From the repository root, start any static HTTP server:

~~~sh
python3 -m http.server 8080
~~~

Open [http://localhost:8080/background-removal-experiment/](http://localhost:8080/background-removal-experiment/). The GitHub Pages copy is available at [https://dragonfiregames.github.io/imagegen/background-removal-experiment/](https://dragonfiregames.github.io/imagegen/background-removal-experiment/).

## Methods

- **AI remove background:** BRIA RMBG-1.4 through the repository's vendored Transformers.js 3.8.1 runtime. It attempts WebGPU FP16 first and can fall back to quantized WebAssembly if Auto is selected.
- **Sampled-color removal:** a CPU-only four-connected flood fill seeded by a clicked background pixel. Tolerance and edge softness control which neighboring pixels are removed. It does not require model downloads.
- **Mask brush:** Restore and Erase paint directly into the alpha mask. Undo restores the previous brush stroke; Reset returns to the first segmentation result.
- **Feather:** blurs the mask edge at preview/export time. The original image pixels are kept; only alpha is changed.

The model is capped to a 1600-pixel longest side for inference. The resulting mask is scaled to the source dimensions for export, which helps keep processing memory use predictable.

## Model and licensing

The default model is [BRIA RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4). Transformers.js downloads the selected ONNX weights from Hugging Face and caches them using the browser cache. The model card states that RMBG-1.4 is licensed for non-commercial use; commercial use requires a separate agreement with BRIA. Review the [model card and license](https://huggingface.co/briaai/RMBG-1.4) before use or redistribution.

The experiment's JavaScript and UI are part of this MIT-licensed repository. The Transformers.js runtime is vendored separately with its upstream license in `vendor/transformers/LICENSE`. Model weights are not included in this repository.

## Privacy and limitations

- The image is decoded, resized for inference, segmented, refined, and exported locally. It is not sent to a background-removal API.
- The browser downloads the model weights from Hugging Face on first AI use. Cached weights may be evicted by the browser.
- The color-key method only removes a connected region similar to the sampled pixel. It works best with a fairly uniform background; it is not a semantic segmenter.
- AI masks are not perfect. Hair, motion blur, transparent objects, shadows, and similarly colored foreground/background regions may need brush cleanup.
- The model has a non-commercial license. The color-key mode has no model-license restriction.

## Upstream references

- [BRIA RMBG-1.4 model card](https://huggingface.co/briaai/RMBG-1.4)
- [Transformers.js documentation](https://huggingface.co/docs/transformers.js)
- [Transformers.js source](https://github.com/huggingface/transformers.js)
