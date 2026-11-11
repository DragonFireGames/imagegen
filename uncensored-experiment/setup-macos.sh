#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="$ROOT/uncensored-experiment"
LOCAL="$ROOT/.local"
ENGINE="$LOCAL/stable-diffusion.cpp"
MODELS="$LOCAL/pony-diffusion-v6-xl"
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
  MODEL_FILE="pony-diffusion-v6-xl-Q4_K_S.gguf"
  echo "Using Q4_K_S weights (about 2.8 GB). This is the lower-memory profile."
else
  MODEL_FILE="pony-diffusion-v6-xl-Q8_0.gguf"
  echo "Using Q8_0 weights (about 4.18 GB) for better fidelity."
fi

MODEL_URL="https://huggingface.co/off-grid-ai/pony-diffusion-v6-xl-GGUF/resolve/main/$MODEL_FILE"
download "$MODEL_URL" "$MODELS/$MODEL_FILE"

cat > "$EXP/config.local.json" <<EOF
{
  "sdCli": ".local/stable-diffusion.cpp/build/bin/sd-cli",
  "model": ".local/pony-diffusion-v6-xl/$MODEL_FILE",
  "offloadToCpu": true
}
EOF

echo
echo "Setup finished."
echo "Start the local UI with:"
echo "  node \"$EXP/server.mjs\""
echo "Then open http://127.0.0.1:8788"
