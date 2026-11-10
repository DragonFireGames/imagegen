# Third-party notices

The application source is MIT-licensed; the following components retain their own upstream licenses.

| Component | Pinned version/source | License | Location |
| --- | --- | --- | --- |
| ONNX Runtime Web | npm onnxruntime-web@1.31.1 | MIT | vendor/onnxruntime-web/LICENSE |
| Transformers.js | npm @huggingface/transformers@3.8.1 | Apache-2.0 | vendor/transformers/LICENSE |
| FLUX.2 Klein tokenizer assets | black-forest-labs/FLUX.2-klein-4B, revision 5e67da950fce4a097bc150c22958a05716994cea | Apache-2.0 | models/qwen3-4b-tokenizer/LICENSE |
| FLUX.2 Klein 4B ONNX weights | cgb/flux2-klein-4b-onnx-webgpu | Apache-2.0, as declared by the publisher | Fetched at runtime; not stored in this repository |

Upstream projects:
- https://github.com/microsoft/onnxruntime
- https://github.com/huggingface/transformers.js
- https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- https://huggingface.co/cgb/flux2-klein-4b-onnx-webgpu

The package-level license files are copied alongside the vendored distributions. The tokenizer license is copied alongside its files. Review upstream notices for any additional attribution requirements.
