#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$ROOT"

echo "Packing ONNX Runtime Web 1.31.1..."
npm pack onnxruntime-web@1.31.1 --silent --pack-destination "$TMP"
mkdir -p "$TMP/ort" vendor/onnxruntime-web
tar -xzf "$TMP/onnxruntime-web-1.31.1.tgz" -C "$TMP/ort"
cp "$TMP/ort/package/dist/ort.webgpu.min.mjs" vendor/onnxruntime-web/
find "$TMP/ort/package/dist" -maxdepth 1 -type f \( -name 'ort-wasm-*.jsep.wasm' -o -name 'ort-wasm-*.jsep.mjs' \) -exec cp {} vendor/onnxruntime-web/ \;
cp "$TMP/ort/package/LICENSE" vendor/onnxruntime-web/LICENSE

echo "Packing Transformers.js 3.8.1..."
npm pack @huggingface/transformers@3.8.1 --silent --pack-destination "$TMP"
mkdir -p "$TMP/transformers" vendor/transformers
tar -xzf "$TMP/huggingface-transformers-3.8.1.tgz" -C "$TMP/transformers"
cp -a "$TMP/transformers/package/dist/." vendor/transformers/
cp "$TMP/transformers/package/LICENSE" vendor/transformers/LICENSE

echo "Vendoring the FLUX.2 tokenizer..."
TOKENIZER_DIR="models/qwen3-4b-tokenizer"
mkdir -p "$TOKENIZER_DIR"
TOKENIZER_BASE="https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/5e67da950fce4a097bc150c22958a05716994cea/tokenizer"
for file in added_tokens.json chat_template.jinja merges.txt special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json; do
  echo "  - $file"
  curl --fail --location --retry 3 --retry-delay 2 "$TOKENIZER_BASE/$file" -o "$TOKENIZER_DIR/$file"
done
curl --fail --location --retry 3 "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/5e67da950fce4a097bc150c22958a05716994cea/LICENSE" -o "$TOKENIZER_DIR/LICENSE"

echo "Vendored file sizes:"
du -sh vendor/onnxruntime-web vendor/transformers "$TOKENIZER_DIR"
