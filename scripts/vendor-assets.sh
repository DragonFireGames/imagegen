#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$ROOT"

echo "Packing ONNX Runtime Web 1.31.1..."
npm pack onnxruntime-web@1.31.1 --silent --pack-destination "$TMP"
mkdir -p "$TMP/ort" vendor/onnxruntime-web noobai-browser-experiment/vendor
tar -xzf "$TMP/onnxruntime-web-1.31.1.tgz" -C "$TMP/ort"
cp "$TMP/ort/package/dist/ort.webgpu.min.mjs" vendor/onnxruntime-web/
# The root FLUX app uses Asyncify. Keep its matching module and WASM binary.
find vendor/onnxruntime-web -maxdepth 1 -type f -name 'ort-wasm-*' -delete
for extension in mjs wasm; do
  file="ort-wasm-simd-threaded.asyncify.$extension"
  if [ ! -f "$TMP/ort/package/dist/$file" ]; then
    echo "Missing required ONNX Runtime WebGPU asset: $file" >&2
    exit 1
  fi
  cp "$TMP/ort/package/dist/$file" vendor/onnxruntime-web/
done

# The NoobAI SDXL experiment uses the official JSPI build. It accepts Blob
# external-data files so OPFS-backed weights do not need one giant JS buffer.
cp "$TMP/ort/package/dist/ort.jspi.bundle.min.mjs" noobai-browser-experiment/vendor/
for extension in mjs wasm; do
  file="ort-wasm-simd-threaded.jspi.$extension"
  if [ ! -f "$TMP/ort/package/dist/$file" ]; then
    echo "Missing required ONNX Runtime JSPI asset: $file" >&2
    exit 1
  fi
  cp "$TMP/ort/package/dist/$file" noobai-browser-experiment/vendor/
done

# Some npm tarballs omit the repository-level license file. Keep the
# license beside both vendored runtime entry points.
if [ -f "$TMP/ort/package/LICENSE" ]; then
  cp "$TMP/ort/package/LICENSE" vendor/onnxruntime-web/LICENSE
else
  curl --fail --location --retry 3 "https://raw.githubusercontent.com/microsoft/onnxruntime/main/LICENSE" -o vendor/onnxruntime-web/LICENSE
fi
cp vendor/onnxruntime-web/LICENSE noobai-browser-experiment/vendor/LICENSE

echo "Packing Transformers.js 3.8.1..."
npm pack @huggingface/transformers@3.8.1 --silent --pack-destination "$TMP"
mkdir -p "$TMP/transformers" vendor/transformers
tar -xzf "$TMP/huggingface-transformers-3.8.1.tgz" -C "$TMP/transformers"
cp -a "$TMP/transformers/package/dist/." vendor/transformers/
cp "$TMP/transformers/package/LICENSE" vendor/transformers/LICENSE
# The experiment is under its own service-worker scope; vendor its tokenizer
# runtime inside that scope so it remains available offline after first load.
mkdir -p noobai-browser-experiment/vendor/transformers
cp "$TMP/transformers/package/dist/transformers.min.js" noobai-browser-experiment/vendor/transformers/
cp "$TMP/transformers/package/LICENSE" noobai-browser-experiment/vendor/transformers/LICENSE

echo "Vendoring the FLUX.2 tokenizer..."
TOKENIZER_DIR="models/qwen3-4b-tokenizer"
mkdir -p "$TOKENIZER_DIR"
TOKENIZER_BASE="https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/5e67da950fce4a097bc150c22958a05716994cea/tokenizer"
for file in added_tokens.json chat_template.jinja merges.txt special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json; do
  echo "  - $file"
  curl --fail --location --retry 3 --retry-delay 2 "$TOKENIZER_BASE/$file" -o "$TOKENIZER_DIR/$file"
done
# The pinned FLUX.2 [klein] 4B model card declares Apache-2.0, but
# that repository revision does not expose a root LICENSE file.
curl --fail --location --retry 3 "https://www.apache.org/licenses/LICENSE-2.0.txt" -o "$TOKENIZER_DIR/LICENSE"

echo "Vendored file sizes:"
du -sh vendor/onnxruntime-web noobai-browser-experiment/vendor vendor/transformers "$TOKENIZER_DIR"
