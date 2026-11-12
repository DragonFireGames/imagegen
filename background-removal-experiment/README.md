# Background Removal Lab

A browser-local image background-removal experiment. It combines BRIA RMBG-2.0, a non-AI color-key tool, and manual alpha-matte cleanup.

## Run

From the repository root, start any static HTTP server:

~~~sh
python3 -m http.server 8080
~~~

Open [http://localhost:8080/background-removal-experiment/](http://localhost:8080/background-removal-experiment/). The GitHub Pages copy is available at [https://dragonfiregames.github.io/imagegen/background-removal-experiment/](https://dragonfiregames.github.io/imagegen/background-removal-experiment/).

## Methods

- **AI remove background:** BRIA RMBG-2.0, using a fixed ONNX export. It runs the FP16-weight graph through WebGPU first and falls back to a quantized INT8 graph through WebAssembly when Auto is selected.
- **Sampled-color removal:** a CPU-only four-connected flood fill seeded by a clicked background pixel. Tolerance and edge softness control which neighboring pixels are removed. It does not require model downloads.
- **Mask brush:** Restore and Erase paint directly into the alpha matte. Undo restores the previous brush stroke; Reset returns to the first segmentation result.
- **Feather:** blurs the matte edge at preview/export time. The original image pixels are kept; only alpha is changed.

The model uses a fixed 1024 × 1024 input. The input is normalized with ImageNet RGB means and standard deviations; the returned alpha matte is bilinearly scaled to the editable image size. The editing canvas is capped to a 1600-pixel longest side to keep brush and flood-fill memory use predictable. PNG exports retain the original image dimensions.

## Model, runtime, and licensing

The app uses [BRIA RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) through the public [baby2008/RMBG-2.0-ONNX fixed export](https://huggingface.co/baby2008/RMBG-2.0-ONNX). Its maintainer reports that the ONNX exports were fixed for standard ONNX Runtime optimizations: the FP16 file stores weights in half precision and casts them to FP32 for computation (about 490 MB), while the INT8 file is about 350 MB. The app loads the FP16 graph with WebGPU and the INT8 graph with WebAssembly. Model bytes are cached with the browser Cache API after the first download.

The export identifies its weights as BRIA RMBG-2.0 under the **Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)** license. Commercial use requires a separate agreement with BRIA. Review the [original model card and license](https://huggingface.co/briaai/RMBG-2.0) and the [fixed ONNX export card](https://huggingface.co/baby2008/RMBG-2.0-ONNX) before use or redistribution.

Inference uses the repository's vendored ONNX Runtime WebGPU build in `vendor/onnxruntime-web/`. Model weights are not included in this repository.

## Privacy and limitations

- The image is decoded, resized for inference, segmented, refined, and exported locally. It is not sent to a background-removal API.
- The browser downloads the model weights from Hugging Face on first AI use. Cached weights may be evicted if browser storage is cleared or under storage pressure.
- The color-key method only removes a connected region similar to the sampled pixel. It works best with a fairly uniform background; it is not a semantic segmenter.
- AI masks are not perfect. Hair, motion blur, transparent objects, shadows, and similarly colored foreground/background regions may need brush cleanup.
- The editing canvas is capped to 1600px on the longest side, and the model itself receives a 1024 × 1024 image. This may limit detail compared with services that process large originals with specialized refinement stages.
- The RMBG-2.0 weights have a non-commercial license. The color-key mode has no model-license restriction.

## Upstream references

- [BRIA RMBG-2.0 model card](https://huggingface.co/briaai/RMBG-2.0)
- [Fixed RMBG-2.0 ONNX export](https://huggingface.co/baby2008/RMBG-2.0-ONNX)
- [ONNX Runtime WebGPU documentation](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
