# Unfiltered Lab — Pony Diffusion V6 XL

This is an isolated local experiment. It does not modify the existing FLUX.2 Klein WebGPU pipeline.

## Why this model

Pony Diffusion V6 XL is an SDXL fine-tune focused on stylized anime, cartoon, furry, and pony art. Its creator reports about 2.6 million training images, with an approximately 1:1:1 balance across safe, questionable, and explicit ratings. Images were captioned with both natural-language captions and tags where available. The creator also states that explicit content involving underage characters was filtered out.

That is a materially different starting point from a model that merely has no safety checker. This model was trained with examples from all three content ratings. It still has learned biases and limitations; no model can guarantee that every prompt will be followed.

## Negative prompting and settings

The UI sends the positive prompt and negative prompt separately to stable-diffusion.cpp. Negative conditioning is real classifier-free guidance, controlled by the CFG slider. The default is 7.0 CFG, Euler a, 25 steps, and clip-skip 2. These are conventional Pony V6 settings; adjust them if the composition or style benefits from different guidance.

Pony uses score and rating tags. Useful tags include:
- Quality: score_9, score_8_up, score_7_up
- Rating: rating_safe, rating_questionable, rating_explicit

The app does not automatically prepend safe or add nsfw to the negative prompt. Those are SFW-oriented prompt recommendations found in some model guides, not a required model-level safety mechanism. The initial negative prompt is only for image quality and anatomy artifacts; users can edit or clear it.

## Runtime

The model is not being represented as a browser WebGPU model. Its GGUF weights run locally through stable-diffusion.cpp, which supports SDXL and Apple's Metal backend. The Node.js bridge serves the page on 127.0.0.1 only and invokes the native CLI with an argument array (no shell interpolation). Prompts and generated images stay on the local machine.

The GitHub Pages page is an informational/setup landing page. Actual generation requires the local server.

## macOS setup

Prerequisites: Xcode Command Line Tools, CMake, Git, curl, and Node.js 18 or newer.

From the repository root:

    ./uncensored-experiment/setup-macos.sh
    node uncensored-experiment/server.mjs

Then open http://127.0.0.1:8788.

The default Q8 profile downloads one GGUF file of about 4.18 GB. On a machine with limited available memory, use:

    MODEL_PROFILE=low-memory ./uncensored-experiment/setup-macos.sh

The Q4_K_S file is about 2.8 GB and trades some fidelity for a smaller working set. Start with 768×768. Increase image area only after a successful generation.

The setup script builds stable-diffusion.cpp with Metal at a pinned upstream commit. Model weights are stored under .local/ and are not committed to Git. The setup script can be rerun safely; completed downloads are reused.

## Controls

- Positive prompt and negative prompt.
- CFG scale 1–12.
- Width/height 512–1536, multiples of 64.
- 10–60 sampling steps.
- Fixed or random seed.
- PNG output and inference logs.

The server permits one generation at a time and binds to loopback only. Do not expose it publicly without adding authentication and origin checks.

## Model files and quantization

The original Pony Diffusion V6 XL checkpoint is distributed as safetensors. For this experiment, we use a conversion prepared for stable-diffusion.cpp:
- Q8_0: about 4.18 GB, better fidelity.
- Q4_K_S: about 2.8 GB, lower memory use.

The conversion repository says these files were converted with stable-diffusion.cpp's own conversion path so that the tensor names load directly. This avoids having to manually assemble separate CLIP and VAE files.

## Licensing

Important distinction: the model is open weights, but it is not OSI-approved open-source software. The original Pony Diffusion V6 XL model is under a modified Fair AI Public License 1.0-SD. The added model terms prohibit inference on websites or applications with monetization without permission; commercial use requires contacting the creator. The base license also prohibits harmful uses including exploiting or harming minors, harassment, and several other harmful activities. Read the original model's terms before redistribution or commercial use.

The quantized GGUF repository's metadata lists CreativeML OpenRAIL-M, but a format conversion does not replace the original model's license. This experiment follows the original model's modified Fair AI license.

- Original Pony Diffusion V6 XL and model card: https://huggingface.co/LyliaEngine/Pony_Diffusion_V6_XL
- Creator's source model page: https://civitai.com/models/257749
- GGUF conversion and stable-diffusion.cpp instructions: https://huggingface.co/off-grid-ai/pony-diffusion-v6-xl-GGUF
- Fair AI Public License 1.0-SD text: https://freedevproject.org/faipl-1.0-sd/
- stable-diffusion.cpp source: https://github.com/leejet/stable-diffusion.cpp

This is an adult-capable model trained on explicit-rated images, but it is not a promise of unrestricted behavior or unrestricted licensing. The model creator reports filtering explicit underage content from its training data; use it lawfully and do not use it to create exploitative or non-consensual material.
