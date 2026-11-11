# Unfiltered Lab — Qwen-Image-2.1

This is an isolated experiment. It does not modify the existing FLUX.2 Klein WebGPU pipeline.

## Model choice

As of November 11, 2026, Qwen-Image-2.1 is at the top of Artificial Analysis's open-weights text-to-image leaderboard (1046 Elo, 95% interval 1034–1058; 6,184 comparisons). FLUX 3 Image Klein and Fibo 2.0 are statistically close behind. Qwen-Image-2.1 is a 7B model and supports native 2K output.

The local Qwen Image inference pipeline ships without a separate safety checker/content-filter stage. A community GGUF distribution labels the model "Uncensored", but its card says the weights are quantized from the original upstream Qwen-Image-2.1. The practical meaning here is **no extra app-side prompt blacklist or output filter**; the original model's learned training behavior still applies. This is not a guarantee that every concept will render exactly as requested.

An independently published test found that ablating the text encoder's refusal direction made essentially no measurable difference to image outputs across 45 sensitive prompts: the image pipeline already had no safety checker, while the 7B denoiser's training still determines what it can draw. For that reason, this experiment uses the supported stock Qwen3-VL text encoder rather than adding an unvalidated "uncensor" patch.

## Negative prompt behavior

Qwen-Image-2.1 supports true classifier-free guidance (CFG). The negative prompt is passed separately to sd-cli with `--negative-prompt`; `--cfg-scale` controls its influence. At guidance scale 1, the negative branch is disabled/has no effect. The UI defaults to 6.0, matching the stable-diffusion.cpp Qwen 2.1 example. Higher guidance costs more compute and can make results worse if pushed too far.

## Why the runtime is native

This model is not being represented as a browser WebGPU model. The 7B DiT, 8B Qwen3-VL text encoder, and VAE are too large to call the existing ONNX WebGPU pipeline without a separate export, operator validation, and memory work. The experiment uses [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), which implements Qwen-Image-2.1, supports GGUF, and has an Apple Metal backend. A small Node.js bridge serves this page only on `127.0.0.1` and invokes the native CLI with argument arrays (no shell interpolation). Prompts and generated images stay on the local machine.

The GitHub Pages page is a setup/research landing page. To generate, run the local server and open its localhost page.

## macOS setup

Prerequisites: Xcode Command Line Tools, CMake, Git, curl, and Node.js 18 or newer. The script builds the Metal version of stable-diffusion.cpp at a pinned commit and downloads the model assets into `.local/`; no model weights are committed to Git.

From the repository root:

```bash
./uncensored-experiment/setup-macos.sh
node uncensored-experiment/server.mjs
```

Then open [http://127.0.0.1:8788](http://127.0.0.1:8788).

The default `quality` profile downloads:
- Qwen-Image-2.1 Q4_K_M diffusion GGUF: 4.60 GB
- Qwen3-VL-8B-Instruct Q4_K_M text encoder: 5.03 GB
- Qwen-Image-2.1 BF16 VAE: 676 MB

That is about 10.3 GB of weights on disk, before temporary inference buffers. A machine with only 9 GB of total unified memory may not have enough headroom, even with CPU offload. For a smaller memory footprint, run:

```bash
MODEL_PROFILE=low-memory ./uncensored-experiment/setup-macos.sh
```

The low-memory profile uses Q2_K diffusion and text-encoder weights; the combined downloads are much smaller, but quantization can reduce prompt and image fidelity. Start with 512×512 or 768×768. Increase resolution only after a successful test.

## Controls

- Positive prompt and negative prompt.
- Guidance scale 1–10. Negative conditioning is effective only above 1.
- Width/height 512–2048, multiples of 32.
- Steps 10–80.
- Fixed or random seed.
- PNG output and raw inference log.

The server permits one generation at a time and binds to loopback only. It does not add authentication because it is not exposed to the network; do not change the bind address or expose it publicly without adding authentication and origin checks.

## Manual engine configuration

If you already have a suitable sd-cli build and model files, create `uncensored-experiment/config.local.json` (this file is ignored by Git):

```json
{
  "sdCli": ".local/stable-diffusion.cpp/build/bin/sd-cli",
  "diffusionModel": ".local/qwen-image-2.1/qwen-image-2.1-UC-Q4_K_M.gguf",
  "textEncoder": ".local/qwen-image-2.1/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
  "vae": ".local/qwen-image-2.1/qwen_image_2.1_vae_bf16.safetensors",
  "offloadToCpu": true
}
```

The paths are relative to the repository root unless absolute paths are used. Environment variables can override them: `SD_CLI`, `QWEN_DIFFUSION_MODEL`, `QWEN_TEXT_ENCODER`, `QWEN_VAE`, and `PORT`.

For Linux or Windows, build [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) with the backend appropriate to your GPU, then use the manual configuration. Only the macOS Metal setup script is provided here.

## Licensing and attribution

"Open source" needs a licensing caveat: Qwen's image model and VAE are **not Apache/MIT open source**. The Qwen-Image-2.1 model and VAE use the Qwen Research License Agreement, which grants non-commercial use only; separate commercial permission is required. The text encoder uses Apache-2.0. Review the terms yourself before use or redistribution.

- [Qwen-Image-2.1 model / license](https://huggingface.co/Qwen/Qwen-Image-2.1)
- [Uncensored GGUF distribution and source provenance](https://huggingface.co/YueFong/Qwen-Image-2.1-Uncensored-GGUF)
- [Qwen3-VL-8B-Instruct GGUF text encoder](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)
- [Qwen VAE file](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/vae/qwen_image_2.1_vae_bf16.safetensors)
- [stable-diffusion.cpp Qwen-Image-2.1 documentation](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md)
- [Artificial Analysis open-weights text-to-image leaderboard](https://artificialanalysis.ai/image/leaderboard/text-to-image/open-weights)
