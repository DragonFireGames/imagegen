# Waifu2x Unlimited

A separate browser-local copy of the **Waifu2x Unlimited** web application, copied from the current `nagadomi/nunif` source. It is an independent experiment and does not share code or runtime state with the FLUX.2 / Illustrious experiments.

The copied application includes the upstream WebGPU and WebAssembly inference backends, model selection, denoising, upscaling, tile processing and shuffling, TTA, alpha-channel handling, image drag-and-drop/paste, stop control, and PNG download.

## Run

From the repository root, start any static HTTP server:

```sh
python3 -m http.server 8080
```

Open [http://localhost:8080/waifu2x-unlimited/](http://localhost:8080/waifu2x-unlimited/). The app requires HTTP(S); it will not work from a `file://` URL.

The GitHub Pages copy is available at [https://dragonfiregames.github.io/imagegen/waifu2x-unlimited/](https://dragonfiregames.github.io/imagegen/waifu2x-unlimited/).

## Model files

The ONNX model weights are intentionally **not committed**: the official archives are hundreds of megabytes, while GitHub Pages has a 1 GB published-site limit. The app expects its model files under this directory's `models/` folder.

For the default `swin_unet_v3 / art` model, download the [2026-08-16 swin_unet_v3 art model archive](https://github.com/nagadomi/nunif/releases/download/0.0.0/waifu2x_pretrained_models_swin_unet_v3_art_20260816.zip) and extract its model contents into `waifu2x-unlimited/models/`.

For the older `swin_unet` and `cunet` model choices (including art-scan and photo models), also download the [2025-05-02 ONNX model archive](https://github.com/nagadomi/nunif/releases/download/0.0.0/waifu2x_onnx_models_20250502.zip) and extract its model contents into the same `models/` directory. Preserve the archive's `models/<architecture>/...` directory structure. The expected paths are generated in `js/models.js`.

The `models/README.md` file in this folder repeats these setup instructions. The archives are published under the upstream project's `0.0.0` GitHub Release.

## External model hosting

You can keep the ONNX files on a separate static host instead of putting them in this repository. In the **Model URL** field, enter the HTTPS URL of the directory containing the model folders, then click **Apply**. For example:

```text
https://raw.githubusercontent.com/USERNAME/MODELS-REPO/BRANCH/models/
```

That directory must contain paths such as:

```text
swin_unet_v3/art/noise0_scale2x.onnx
utils/create_seam_blending_filter.onnx
```

The app appends the required model paths to the URL automatically. Use a host that serves the individual extracted ONNX files and sends permissive CORS headers. A GitHub release ZIP URL is not a model directory: extract the archive and host the resulting folder structure first. The URL setting is saved in this browser. Leave it as `models/` to use files hosted alongside the app.

## Runtime and privacy

- **Auto** selects WebGPU when available and falls back to WebAssembly; the backend can also be selected manually.
- Image processing runs in the browser. The input image is not uploaded to an inference server.
- The upstream page pins ONNX Runtime Web to a development build served by jsDelivr. Model files can be served locally from this app's `models/` folder or from the external directory configured in **Model URL**.
- Very large images can still exceed browser or GPU memory limits; tile size controls the amount processed at once.

## Upstream source

- Project: [nagadomi/nunif](https://github.com/nagadomi/nunif)
- Copied directory: `waifu2x/unlimited_waifu2x/public_html/`
- Upstream revision: `b1548e8ff08758f336cdab14443647318ef5e0e9`
- Upstream license: MIT, preserved in [LICENSE-nunif](./LICENSE-nunif)

The app files are kept close to upstream to make later comparisons and experiments straightforward. The small Imagegen integration consists of the back-link and local favicon.
