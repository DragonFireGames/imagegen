#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="$ROOT/uncensored-experiment"
LOCAL="$ROOT/.local"
ENGINE="$LOCAL/stable-diffusion.cpp"
MODELS="$LOCAL/qwen-image-2.1"
SD_CPP_COMMIT="32d516c0197ed5ca31305b8575e14eb1451fad33"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script builds the Metal backend for macOS only." >&2
  echo "On Linux/Windows, build stable-diffusion.cpp with your GPU backend and edit config.local.json." >&2
  exit 1
fi

for tool in git cmake curl; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing prerequisite: $tool. Install Xcode Command Line Tools and CMake, then rerun." >&2
    exit 1
  fi
done

mkdir -p "$LOCAL" "$MODELS"

if [[ ! -d "$ENGINE/.git" ]]; then
  git clone --recursive https://github.com/leejet/stable-diffusion.cpp.git "$ENGINE"
else
  git -C "$ENGINE" fetch origin
fi

git -C "$ENGINE" checkout "$SD_CPP_COMMIT"
git -C "$ENGINE" submodule update --init --recursive

echo "Building stable-diffusion.cpp with the Apple Metal backend..."
cmake -S "$ENGINE" -B "$ENGINE/build" \
  -DSD_METAL=ON \
  -DSD_BUILD_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$ENGINE/build" --config Release --parallel "${BUILD_JOBS:-2}" --target sd-cli

SD_CLI="$ENGINE/build/bin/sd-cli"
if [[ ! -x "$SD_CLI" ]]; then
  echo "Build completed but $SD_CLI was not found." >&2
  exit 1
fi

download() {
  local url="$1"
  local destination="$2"
  if [[ -f "$destination" ]]; then
    echo "Already downloaded: $(basename "$destination")"
    return
  fi
  local partial="${destination}.partial"
  echo "Downloading $(basename "$destination")..."
  curl --fail --location --retry 3 --retry-delay 2 --continue-at - "$url" -o "$partial"
  mv "$partial" "$destination"
}

if [[ "${MODEL_PROFILE:-quality}" == "low-memory" ]]; then
  DIFFUSION_FILE="qwen_image_2.1-Q2_K.gguf"
  TEXT_FILE="Qwen3-VL-8B-Instruct-Q2_K.gguf"
  DIFFUSION_URL="https://huggingface.co/leejet/Qwen-Image-2.1-GGUF/resolve/main/$DIFFUSION_FILE"
  TEXT_URL="https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/$TEXT_FILE"
  echo "Using Q2_K diffusion and text-encoder weights. This saves memory but reduces image/prompt fidelity."
else
  DIFFUSION_FILE="qwen-image-2.1-UC-Q4_K_M.gguf"
  TEXT_FILE="Qwen3VL-8B-Instruct-Q4_K_M.gguf"
  DIFFUSION_URL="https://huggingface.co/YueFong/Qwen-Image-2.1-Uncensored-GGUF/resolve/main/$DIFFUSION_FILE"
  TEXT_URL="https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/$TEXT_FILE"
  echo "Using Q4_K_M diffusion and text-encoder weights for better fidelity."
fi

download "$DIFFUSION_URL" "$MODELS/$DIFFUSION_FILE"
download "$TEXT_URL" "$MODELS/$TEXT_FILE"
download \
  "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors" \
  "$MODELS/qwen_image_2.1_vae_bf16.safetensors"

cat > "$EXP/config.local.json" <<EOF
{
  "sdCli": ".local/stable-diffusion.cpp/build/bin/sd-cli",
  "diffusionModel": ".local/qwen-image-2.1/$DIFFUSION_FILE",
  "textEncoder": ".local/qwen-image-2.1/$TEXT_FILE",
  "vae": ".local/qwen-image-2.1/qwen_image_2.1_vae_bf16.safetensors",
  "offloadToCpu": true
}
EOF

echo
echo "Setup finished."
echo "Start the local UI with:"
echo "  node \"$EXP/server.mjs\""
echo "Then open http://127.0.0.1:8788"
