# Model assets

This directory contains the small tokenizer/configuration files bundled with the application. The large ONNX model weights are not stored here.

The tokenizer files are copied from black-forest-labs/FLUX.2-klein-4B/tokenizer at revision 5e67da950fce4a097bc150c22958a05716994cea by scripts/vendor-assets.sh.

The ONNX model graphs and their sharded external data files are fetched at runtime from:

https://huggingface.co/cgb/flux2-klein-4b-onnx-webgpu/resolve/main/

The model publisher declares Apache-2.0. See that repository's README and LICENSE for model terms.
