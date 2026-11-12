# Pixel Art Studio

Local-first pixel art editor experiment for DragonFireGames Imagegen.

Open: https://dragonfiregames.github.io/imagegen/pixel-art-studio/

## Current features
- Pixel-perfect canvas: pencil, eraser, bucket fill, eyedropper, line, rectangle, ellipse.
- Canvas resize, zoom, brush size, color picker, custom palette, undo/redo.
- Import PNG, WebP, JPEG, or GIF (first frame); export transparent PNG.
- Local pixelation, palette quantization, Floyd–Steinberg dithering, outline, and Game Boy / NES / PICO-8 / 1-bit presets.
- AI handoff to the existing local Illustrious browser generator. The prompt is passed into the generator URL. Generate at 512px, download the PNG, then import and reduce to the requested sprite size.
- AI edit handoff to the existing local brush-infill experiment. Import the source image there, edit, download, and bring it back here.

## Important AI distinction
Sprite Fusion advertises a purpose-built pixel-art model for exact-size generation, prompt-based sprite editing, style matching, 8-direction sets, and animation. This project does **not** yet have an equivalent dedicated local pixel-art model. The current AI workflow reuses the local SDXL/Illustrious model already in this repository, then uses explicit pixel-art conversion and manual editing. Pixel processors are deterministic local image operations, not AI.

A dedicated local pixel-art model and true image-conditioned prompt editing are the next research/integration tasks. The app does not claim that the existing SDXL generator is a specialized pixel-art model.

No generation server, API key, account, or app-side upload is used. Model weights remain managed by the sibling experiments and are downloaded/cached separately.
