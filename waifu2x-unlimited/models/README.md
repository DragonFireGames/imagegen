# Model files (not tracked)

Place extracted Waifu2x ONNX model files in this folder. Preserve the `models/<architecture>/<style>/<method>.onnx` layout expected by `../js/models.js`.

Required for the default model:
- [swin_unet_v3 art, 2026-08-16](https://github.com/nagadomi/nunif/releases/download/0.0.0/waifu2x_pretrained_models_swin_unet_v3_art_20260816.zip)

For the legacy model choices:
- [ONNX model pack, 2025-05-02](https://github.com/nagadomi/nunif/releases/download/0.0.0/waifu2x_onnx_models_20250502.zip)

Extract the model contents here, not into a nested `public_html/` directory. The archives are intentionally kept out of Git.

## External hosting

Instead of placing files here, host the extracted `models/` directory on a static server that allows CORS. In the app's **Model URL** field, enter the URL to that directory (the directory containing `swin_unet_v3/` and `utils/`), then click **Apply**. The app constructs each ONNX file URL from that root. Do not enter the URL of the ZIP archive itself.
