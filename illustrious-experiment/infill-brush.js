const sourceFile = document.querySelector("#sourceImageFile");
const sourcePreview = document.querySelector("#sourcePreview");
const paintOverlay = document.querySelector("#paintOverlay");
const canvasStack = document.querySelector("#canvasStack");
const canvasEmpty = document.querySelector("#canvasEmpty");
const runButton = document.querySelector("#runInfill");
const status = document.querySelector("#generationStatus");
const brushSize = document.querySelector("#brushSize");
const brushSizeValue = document.querySelector("#brushSizeValue");
const brushSoftness = document.querySelector("#brushSoftness");
const brushSoftnessValue = document.querySelector("#brushSoftnessValue");
const strengthInput = document.querySelector("#infillStrength");
const strengthValue = document.querySelector("#infillStrengthValue");
const sourceCanvas = document.createElement("canvas");
const maskCanvas = document.createElement("canvas");
let sourceBitmap = null;
let brushMode = "paint";
let pointerActive = false;
let previousPoint = null;
let imageWidth = 512;
let imageHeight = 512;

function setStatus(message, type = "info") {
  status.textContent = message;
  status.dataset.type = type;
}

function chooseModelSize(width, height) {
  const ratio = width / height;
  let best = null;
  for (let w = 512; w <= 1024; w += 64) {
    for (let h = 512; h <= 1024; h += 64) {
      const error = Math.abs(Math.log((w / h) / ratio));
      const score = error * 1000000 - (w * h);
      if (!best || score < best.score) best = { width: w, height: h, score };
    }
  }
  return { width: best.width, height: best.height };
}

function drawSource() {
  const context = sourceCanvas.getContext("2d");
  const scale = Math.max(imageWidth / sourceBitmap.width, imageHeight / sourceBitmap.height);
  const drawWidth = sourceBitmap.width * scale;
  const drawHeight = sourceBitmap.height * scale;
  const offsetX = (imageWidth - drawWidth) / 2;
  const offsetY = (imageHeight - drawHeight) / 2;
  context.clearRect(0, 0, imageWidth, imageHeight);
  context.drawImage(sourceBitmap, offsetX, offsetY, drawWidth, drawHeight);
}

function renderPreview() {
  const sourceContext = sourcePreview.getContext("2d");
  sourceContext.clearRect(0, 0, imageWidth, imageHeight);
  sourceContext.drawImage(sourceCanvas, 0, 0);

  const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true });
  const mask = maskContext.getImageData(0, 0, imageWidth, imageHeight);
  const overlayContext = paintOverlay.getContext("2d");
  const overlay = overlayContext.createImageData(imageWidth, imageHeight);
  for (let i = 0; i < mask.data.length; i += 4) {
    const alpha = mask.data[i + 3];
    if (!alpha) continue;
    overlay.data[i] = 248;
    overlay.data[i + 1] = 67;
    overlay.data[i + 2] = 132;
    overlay.data[i + 3] = Math.round(alpha * 0.48);
  }
  overlayContext.putImageData(overlay, 0, 0);
}

function setBrushMode(mode) {
  brushMode = mode;
  document.querySelector("#brushTool").classList.toggle("selected", mode === "paint");
  document.querySelector("#eraseTool").classList.toggle("selected", mode === "erase");
  paintOverlay.style.cursor = mode === "paint" ? "crosshair" : "cell";
}

function brushStamp(context, x, y) {
  const radius = Number(brushSize.value) / 2;
  const softness = Number(brushSoftness.value) / 100;
  context.save();
  context.globalCompositeOperation = brushMode === "erase" ? "destination-out" : "source-over";
  if (softness >= 0.999) {
    context.fillStyle = "#fff";
  } else {
    const inner = radius * (1 - softness);
    const gradient = context.createRadialGradient(x, y, inner, x, y, radius);
    gradient.addColorStop(0, "rgba(255,255,255,1)");
    if (inner > 0) gradient.addColorStop(0.999, "rgba(255,255,255,1)");
    gradient.addColorStop(1, "rgba(255,255,255,0)");
    context.fillStyle = gradient;
  }
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  context.fill();
  context.restore();
}

function paintSegment(from, to) {
  const context = maskCanvas.getContext("2d");
  const distance = Math.hypot(to.x - from.x, to.y - from.y);
  const spacing = Math.max(1, Number(brushSize.value) * 0.18);
  const steps = Math.max(1, Math.ceil(distance / spacing));
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    brushStamp(context, from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t);
  }
  renderPreview();
}

function pointFromEvent(event) {
  const rect = paintOverlay.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(imageWidth, (event.clientX - rect.left) * imageWidth / rect.width)),
    y: Math.max(0, Math.min(imageHeight, (event.clientY - rect.top) * imageHeight / rect.height))
  };
}

paintOverlay.addEventListener("pointerdown", event => {
  if (!sourceBitmap || pointerActive) return;
  event.preventDefault();
  pointerActive = true;
  previousPoint = pointFromEvent(event);
  paintOverlay.setPointerCapture(event.pointerId);
  paintSegment(previousPoint, previousPoint);
});
paintOverlay.addEventListener("pointermove", event => {
  if (!pointerActive) return;
  event.preventDefault();
  const point = pointFromEvent(event);
  paintSegment(previousPoint, point);
  previousPoint = point;
});
function stopPainting() {
  pointerActive = false;
  previousPoint = null;
}
paintOverlay.addEventListener("pointerup", stopPainting);
paintOverlay.addEventListener("pointercancel", stopPainting);
paintOverlay.addEventListener("lostpointercapture", stopPainting);

sourceFile.addEventListener("change", async () => {
  const file = sourceFile.files?.[0];
  if (!file) return;
  try {
    const nextBitmap = await createImageBitmap(file);
    sourceBitmap?.close?.();
    sourceBitmap = nextBitmap;
    const dimensions = chooseModelSize(sourceBitmap.width, sourceBitmap.height);
    imageWidth = dimensions.width;
    imageHeight = dimensions.height;
    sourceCanvas.width = imageWidth;
    sourceCanvas.height = imageHeight;
    maskCanvas.width = imageWidth;
    maskCanvas.height = imageHeight;
    sourcePreview.width = imageWidth;
    sourcePreview.height = imageHeight;
    paintOverlay.width = imageWidth;
    paintOverlay.height = imageHeight;
    drawSource();
    renderPreview();
    document.querySelector("#generationWidth").value = imageWidth;
    document.querySelector("#generationHeight").value = imageHeight;
    document.querySelector("#resolutionInfo").textContent =
      sourceBitmap.width + "×" + sourceBitmap.height + " → " + imageWidth + "×" + imageHeight +
      " · centered crop to the closest SDXL size";
    document.querySelector("#canvasDimensions").textContent = imageWidth + "×" + imageHeight;
    canvasEmpty.hidden = true;
    canvasStack.hidden = false;
    runButton.disabled = false;
    document.querySelector("#resultPanel").hidden = true;
    setStatus("Image loaded. Paint the area to replace, then describe the result.");
  } catch (error) {
    setStatus("Could not open this image: " + (error?.message || String(error)), "error");
  }
});

document.querySelector("#brushTool").addEventListener("click", () => setBrushMode("paint"));
document.querySelector("#eraseTool").addEventListener("click", () => setBrushMode("erase"));
document.querySelector("#clearMask").addEventListener("click", () => {
  if (!sourceBitmap) return;
  maskCanvas.getContext("2d").clearRect(0, 0, imageWidth, imageHeight);
  renderPreview();
});
document.querySelector("#invertMask").addEventListener("click", () => {
  if (!sourceBitmap) return;
  const context = maskCanvas.getContext("2d", { willReadFrequently: true });
  const image = context.getImageData(0, 0, imageWidth, imageHeight);
  for (let i = 0; i < image.data.length; i += 4) {
    const alpha = 255 - image.data[i + 3];
    image.data[i] = image.data[i + 1] = image.data[i + 2] = 255;
    image.data[i + 3] = alpha;
  }
  context.putImageData(image, 0, 0);
  renderPreview();
});

brushSize.addEventListener("input", () => brushSizeValue.textContent = brushSize.value + " px");
brushSoftness.addEventListener("input", () => brushSoftnessValue.textContent = brushSoftness.value + "%");
strengthInput.addEventListener("input", () => strengthValue.textContent = Number(strengthInput.value).toFixed(2));

function hasPaintedMask() {
  const pixels = maskCanvas.getContext("2d", { willReadFrequently: true })
    .getImageData(0, 0, imageWidth, imageHeight).data;
  for (let i = 3; i < pixels.length; i += 4) if (pixels[i] > 8) return true;
  return false;
}

window.__illustriousInfill = {
  getPayload() {
    if (!sourceBitmap) throw new Error("Choose a source image first.");
    if (!hasPaintedMask()) throw new Error("Paint the area you want to replace first.");
    return {
      sourceCanvas,
      maskCanvas,
      strength: Number(strengthInput.value)
    };
  }
};

runButton.addEventListener("click", () => {
  try {
    window.__illustriousInfill.getPayload();
    document.querySelector("#generateImage").click();
  } catch (error) {
    setStatus(error?.message || String(error), "error");
  }
});

document.querySelector("#generationStatus").addEventListener("DOMSubtreeModified", () => {
  const output = document.querySelector("#generatedImage");
  if (!output.hidden && output.src) {
    document.querySelector("#resultImage").src = output.src;
    document.querySelector("#resultPanel").hidden = false;
    document.querySelector("#resultDownload").href = output.src;
    document.querySelector("#resultDownload").download = "illustrious-infill.png";
    runButton.disabled = false;
  }
});
document.querySelector("#downloadGeneratedImage").addEventListener("click", event => {
  const source = document.querySelector("#generatedImage");
  if (!source.src) return;
  event.preventDefault();
  const link = document.createElement("a");
  link.href = source.src;
  link.download = "illustrious-infill.png";
  link.click();
});
document.querySelector("#generationStatus").addEventListener("change", () => {
  runButton.disabled = false;
});
