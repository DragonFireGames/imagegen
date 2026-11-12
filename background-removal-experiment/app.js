import { pipeline, RawImage, env } from "../vendor/transformers/transformers.js";

const MODEL_ID = "briaai/RMBG-1.4";
const MAX_PROCESS_SIDE = 1600;
const $ = (id) => document.getElementById(id);

const fileInput = $("file-input");
const dropzone = $("dropzone");
const sourcePreview = $("source-preview");
const sourceFrame = $("source-frame");
const sourceEmpty = $("source-empty");
const outputEmpty = $("output-empty");
const resultCanvas = $("result-canvas");
const resultContext = resultCanvas.getContext("2d");
const processCanvas = document.createElement("canvas");
const processContext = processCanvas.getContext("2d", { willReadFrequently: true });
const maskCanvas = document.createElement("canvas");
const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true });

let sourceFile = null;
let sourceBitmap = null;
let sourceUrl = null;
let sourceWidth = 0;
let sourceHeight = 0;
let processWidth = 0;
let processHeight = 0;
let maskData = null;
let baseMaskData = null;
let maskReady = false;
let undoData = null;
let brushMode = "none";
let lastBrushPoint = null;
let painting = false;
let pickingColor = false;
let segmenters = {};

env.useBrowserCache = true;
env.useWasmCache = true;
env.cacheKey = "imagegen-background-removal-v1";
env.backends.onnx.wasm.wasmPaths = new URL("../vendor/transformers/", import.meta.url).href;
env.backends.onnx.wasm.numThreads = 1;

function setProgress(label, percent, detail) {
  $("progress-label").textContent = label;
  $("progress-label").classList.remove("error");
  $("progress-bar").style.width = Math.max(0, Math.min(100, percent || 0)) + "%";
  $("progress-detail").textContent = detail || "—";
}

function setStatus(message) {
  $("workspace-status").textContent = message;
  $("runtime-log").textContent = message;
}

function setBusy(busy) {
  $("ai-remove").disabled = busy || !sourceFile;
  $("pick-background").disabled = busy || !sourceFile;
  $("choose-image").disabled = busy;
  fileInput.disabled = busy;
}

function setMaskControls(enabled) {
  $("brush-view").disabled = !enabled;
  $("brush-restore").disabled = !enabled;
  $("brush-erase").disabled = !enabled;
  $("brush-size").disabled = !enabled;
  $("feather").disabled = !enabled;
  $("undo").disabled = !enabled || !undoData;
  $("reset-mask").disabled = !enabled;
  $("download").disabled = !enabled;
  updateBrushButtons();
}

function updateBrushButtons() {
  for (const button of document.querySelectorAll("[data-brush]")) {
    button.classList.toggle("active", button.dataset.brush === brushMode);
  }
  resultCanvas.classList.toggle("painting", brushMode !== "none");
  $("brush-tip").textContent = brushMode === "restore"
    ? "Paint to bring the original image back."
    : brushMode === "erase"
      ? "Paint to make more of the image transparent."
      : "Mask edits affect the exported PNG.";
}

function updateSliderLabels() {
  $("brush-size-value").textContent = $("brush-size").value + " px";
  $("feather-value").textContent = $("feather").value + " px";
  $("tolerance-value").textContent = $("tolerance").value;
  $("color-softness-value").textContent = $("color-softness").value;
}

function drawMask(data) {
  if (!data || !maskCanvas.width || !maskCanvas.height) return;
  const image = maskContext.createImageData(maskCanvas.width, maskCanvas.height);
  for (let i = 0; i < data.length; i++) {
    const offset = i * 4;
    const value = data[i];
    image.data[offset] = value;
    image.data[offset + 1] = value;
    image.data[offset + 2] = value;
    image.data[offset + 3] = 255;
  }
  maskContext.putImageData(image, 0, 0);
}

function renderOutput() {
  if (!sourceBitmap || !maskReady || !maskData) return;
  resultCanvas.width = sourceWidth;
  resultCanvas.height = sourceHeight;
  resultContext.clearRect(0, 0, sourceWidth, sourceHeight);
  resultContext.globalCompositeOperation = "source-over";
  resultContext.filter = "none";
  resultContext.imageSmoothingEnabled = true;
  resultContext.imageSmoothingQuality = "high";
  resultContext.drawImage(sourceBitmap, 0, 0, sourceWidth, sourceHeight);
  resultContext.globalCompositeOperation = "destination-in";
  const feather = Number($("feather").value);
  resultContext.filter = feather > 0 ? "blur(" + feather + "px)" : "none";
  resultContext.drawImage(maskCanvas, 0, 0, sourceWidth, sourceHeight);
  resultContext.filter = "none";
  resultContext.globalCompositeOperation = "source-over";
  resultCanvas.hidden = false;
  outputEmpty.hidden = true;
  $("output-dimensions").textContent = sourceWidth + " × " + sourceHeight;
}

function commitMask(data, setBase) {
  maskData = new Uint8ClampedArray(data);
  maskReady = true;
  if (setBase || !baseMaskData) baseMaskData = new Uint8ClampedArray(maskData);
  undoData = null;
  drawMask(maskData);
  setMaskControls(true);
  renderOutput();
}

async function loadImage(file) {
  if (!file || !file.type || !file.type.startsWith("image/")) {
    setStatus("Choose a supported image file.");
    return;
  }

  setBusy(true);
  try {
    const bitmap = await createImageBitmap(file);
    if (sourceBitmap) sourceBitmap.close();
    sourceBitmap = bitmap;
    sourceFile = file;
    sourceWidth = bitmap.width;
    sourceHeight = bitmap.height;

    if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    sourceUrl = URL.createObjectURL(file);
    sourcePreview.src = sourceUrl;
    sourcePreview.hidden = false;
    sourceEmpty.hidden = true;
    sourcePreview.alt = file.name || "Original image";

    const scale = Math.min(1, MAX_PROCESS_SIDE / Math.max(sourceWidth, sourceHeight));
    processWidth = Math.max(1, Math.round(sourceWidth * scale));
    processHeight = Math.max(1, Math.round(sourceHeight * scale));
    processCanvas.width = processWidth;
    processCanvas.height = processHeight;
    processContext.clearRect(0, 0, processWidth, processHeight);
    processContext.drawImage(bitmap, 0, 0, processWidth, processHeight);

    maskCanvas.width = processWidth;
    maskCanvas.height = processHeight;
    maskData = null;
    baseMaskData = null;
    maskReady = false;
    undoData = null;
    pickingColor = false;
    brushMode = "none";
    sourceFrame.classList.remove("pick-mode");
    resultCanvas.hidden = true;
    outputEmpty.hidden = false;
    $("original-dimensions").textContent = sourceWidth + " × " + sourceHeight;
    $("output-dimensions").textContent = "—";
    $("file-meta").textContent = (file.name || "Image") + " · " + sourceWidth + " × " + sourceHeight +
      " · " + (file.size / 1024 / 1024).toFixed(2) + " MiB";
    $("source-tip").textContent = "The source stays unchanged.";
    $("model-status").textContent = "Model not loaded";
    $("model-status").classList.remove("ready");
    $("pick-background").textContent = "Pick background color from image";
    $("pick-hint").textContent = "Click the background in the Original panel. Connected areas of a similar color will be removed.";
    setMaskControls(false);
    setProgress("Image loaded", 0, sourceWidth + " × " + sourceHeight);
    setStatus("Image ready. Run AI removal or sample a background color.");
    $("runtime-dot").classList.add("ready");
  } catch (error) {
    console.error(error);
    setStatus("Could not decode this image: " + (error.message || error));
    setProgress("Image load failed", 0, "");
    $("progress-label").classList.add("error");
  } finally {
    setBusy(false);
  }
}

function updateModelProgress(event) {
  if (!event) return;
  const file = event.file ? event.file.split("/").pop() : "model files";
  if (event.status === "progress" && Number.isFinite(event.progress)) {
    setProgress("Downloading " + file, Math.min(95, event.progress), Math.round(event.progress) + "%");
  } else if (event.status === "done") {
    setProgress("Loaded " + file, 95, "Preparing runtime");
  } else if (event.status === "initiate") {
    setProgress("Fetching " + file, 5, "");
  } else if (event.status === "ready") {
    setProgress("Preparing model", 95, "");
  } else if (event.status) {
    setProgress(String(event.status), 5, file);
  }
}

async function getSegmenter(device) {
  if (segmenters[device]) return segmenters[device];
  const dtype = device === "webgpu" ? "fp16" : "q8";
  setProgress("Loading RMBG-1.4", 2, device.toUpperCase() + " · " + dtype.toUpperCase());
  setStatus("Loading the " + device.toUpperCase() + " background-removal model. The first run downloads model weights.");
  const instance = await pipeline("image-segmentation", MODEL_ID, {
    device,
    dtype,
    progress_callback: updateModelProgress
  });
  segmenters[device] = instance;
  return instance;
}

function findForegroundSegment(segments) {
  if (!Array.isArray(segments) || !segments.length) return null;
  const foreground = segments.find((item) => /foreground|subject|object/i.test(item.label || ""));
  if (foreground) return foreground;
  const nonBackground = segments.find((item) => !/background/i.test(item.label || ""));
  return nonBackground || segments[0];
}

function readMask(segment) {
  const mask = segment && segment.mask;
  if (!mask || !mask.data || !mask.width || !mask.height) {
    throw new Error("The model returned a segmentation without a readable mask.");
  }
  const output = new Uint8ClampedArray(processWidth * processHeight);
  const channels = mask.channels || 1;
  const data = mask.data;
  for (let y = 0; y < processHeight; y++) {
    const sy = Math.min(mask.height - 1, Math.floor(y * mask.height / processHeight));
    for (let x = 0; x < processWidth; x++) {
      const sx = Math.min(mask.width - 1, Math.floor(x * mask.width / processWidth));
      const sourceIndex = (sy * mask.width + sx) * channels;
      let value = Number(data[sourceIndex]);
      if (!Number.isFinite(value)) value = 0;
      if (value >= 0 && value <= 1) value *= 255;
      output[y * processWidth + x] = Math.max(0, Math.min(255, Math.round(value)));
    }
  }
  return output;
}

async function removeWithAI() {
  if (!sourceFile || !sourceBitmap) return;
  setBusy(true);
  brushMode = "none";
  updateBrushButtons();
  $("pick-background").textContent = "Pick background color from image";
  pickingColor = false;
  sourceFrame.classList.remove("pick-mode");
  $("model-status").textContent = "Loading model";
  $("model-status").classList.remove("ready");

  const requested = $("backend").value;
  const candidates = requested === "auto"
    ? (navigator.gpu ? ["webgpu", "wasm"] : ["wasm"])
    : [requested];
  let lastError = null;

  try {
    const rawImage = RawImage.fromCanvas(processCanvas);
    for (let i = 0; i < candidates.length; i++) {
      const device = candidates[i];
      try {
        if (device === "webgpu" && !navigator.gpu) {
          throw new Error("WebGPU is unavailable in this browser.");
        }
        const model = await getSegmenter(device);
        setProgress("Segmenting foreground", 97, device.toUpperCase());
        setStatus("Running foreground segmentation locally on " + device.toUpperCase() + "…");
        const segments = await model(rawImage, { threshold: 0, mask_threshold: 0.5 });
        const selected = findForegroundSegment(segments);
        if (!selected) throw new Error("The segmentation model did not return a foreground mask.");
        const nextMask = readMask(selected);
        let visible = 0;
        for (let p = 0; p < nextMask.length; p++) if (nextMask[p] > 0) visible++;
        if (!visible) throw new Error("The model returned an empty foreground mask.");
        commitMask(nextMask, true);
        $("model-status").textContent = "Loaded · " + device.toUpperCase();
        $("model-status").classList.add("ready");
        setProgress("Background removed", 100, device.toUpperCase());
        setStatus("AI cutout complete. Use Restore and Erase to refine the mask.");
        $("source-tip").textContent = "Use the mask tools to clean up the cutout.";
        return;
      } catch (error) {
        lastError = error;
        if (device === "webgpu" && requested === "auto" && i + 1 < candidates.length) {
          setStatus("WebGPU failed; retrying with WebAssembly. " + (error.message || error));
          delete segmenters.webgpu;
          continue;
        }
        throw error;
      }
    }
    throw lastError || new Error("No background-removal backend was available.");
  } catch (error) {
    console.error(error);
    setProgress("AI removal failed", 0, "");
    $("progress-label").classList.add("error");
    setStatus("AI removal failed: " + (error.message || error));
    $("model-status").textContent = "Model error";
  } finally {
    setBusy(false);
  }
}

function colorDistance(data, index, target) {
  const offset = index * 4;
  const dr = data[offset] - target[0];
  const dg = data[offset + 1] - target[1];
  const db = data[offset + 2] - target[2];
  return Math.sqrt(dr * dr + dg * dg + db * db);
}

function removeConnectedColor(seedX, seedY) {
  const width = processWidth;
  const height = processHeight;
  const count = width * height;
  const image = processContext.getImageData(0, 0, width, height).data;
  const seedIndex = seedY * width + seedX;
  const seedOffset = seedIndex * 4;
  const target = [image[seedOffset], image[seedOffset + 1], image[seedOffset + 2]];
  const tolerance = Number($("tolerance").value);
  const softness = Number($("color-softness").value);
  const limit = tolerance + softness;
  const visited = new Uint8Array(count);
  const queue = new Int32Array(count);
  const nextMask = maskReady && maskData
    ? new Uint8ClampedArray(maskData)
    : new Uint8ClampedArray(count).fill(255);

  if (!maskReady) baseMaskData = new Uint8ClampedArray(count).fill(255);

  let head = 0;
  let tail = 0;
  queue[tail++] = seedIndex;
  visited[seedIndex] = 1;
  let removed = 0;

  while (head < tail) {
    const index = queue[head++];
    const distance = colorDistance(image, index, target);
    if (distance > limit) continue;

    const alpha = softness > 0
      ? Math.max(0, Math.min(255, Math.round((distance - tolerance) / softness * 255)))
      : (distance <= tolerance ? 0 : 255);
    nextMask[index] = Math.min(nextMask[index], alpha);
    if (alpha < 255) removed++;

    const x = index % width;
    const y = Math.floor(index / width);
    if (x > 0) {
      const neighbor = index - 1;
      if (!visited[neighbor]) { visited[neighbor] = 1; queue[tail++] = neighbor; }
    }
    if (x + 1 < width) {
      const neighbor = index + 1;
      if (!visited[neighbor]) { visited[neighbor] = 1; queue[tail++] = neighbor; }
    }
    if (y > 0) {
      const neighbor = index - width;
      if (!visited[neighbor]) { visited[neighbor] = 1; queue[tail++] = neighbor; }
    }
    if (y + 1 < height) {
      const neighbor = index + width;
      if (!visited[neighbor]) { visited[neighbor] = 1; queue[tail++] = neighbor; }
    }
  }

  if (!removed) {
    setStatus("No connected area matched that color. Increase Color tolerance or pick a flatter part of the background.");
    return;
  }

  commitMask(nextMask, false);
  $("model-status").textContent = "Color key · no model";
  $("model-status").classList.remove("ready");
  setProgress("Color background removed", 100, removed.toLocaleString() + " pixels");
  setStatus("Removed a connected region near RGB(" + target.join(", ") + "). Pick another area to remove an additional color.");
}

function toggleColorPick() {
  if (!sourceFile) return;
  pickingColor = !pickingColor;
  sourceFrame.classList.toggle("pick-mode", pickingColor);
  $("pick-background").textContent = pickingColor ? "Cancel color picking" : "Pick background color from image";
  $("pick-hint").textContent = pickingColor
    ? "Now click a background area in the Original panel. The color-connected region will be removed."
    : "Click the background in the Original panel. Connected areas of a similar color will be removed.";
  setStatus(pickingColor ? "Click the background in the Original panel to sample its color." : "Color picking cancelled.");
}

function sampleBackground(event) {
  if (!pickingColor || !sourceFile) return;
  const rect = sourcePreview.getBoundingClientRect();
  const x = Math.floor((event.clientX - rect.left) / rect.width * processWidth);
  const y = Math.floor((event.clientY - rect.top) / rect.height * processHeight);
  if (x < 0 || y < 0 || x >= processWidth || y >= processHeight) return;
  pickingColor = false;
  sourceFrame.classList.remove("pick-mode");
  $("pick-background").textContent = "Pick another background color";
  removeConnectedColor(x, y);
}

function getBrushPoint(event) {
  const rect = resultCanvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  return {
    x: (event.clientX - rect.left) / rect.width * maskCanvas.width,
    y: (event.clientY - rect.top) / rect.height * maskCanvas.height,
    scale: maskCanvas.width / rect.width
  };
}

function startBrush(event) {
  if (!maskReady || brushMode === "none") return;
  const point = getBrushPoint(event);
  if (!point) return;
  event.preventDefault();
  painting = true;
  lastBrushPoint = point;
  undoData = maskContext.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
  $("undo").disabled = false;
  try { resultCanvas.setPointerCapture(event.pointerId); } catch (error) {}
  maskContext.save();
  maskContext.globalCompositeOperation = "source-over";
  maskContext.strokeStyle = brushMode === "restore" ? "#ffffff" : "#000000";
  maskContext.fillStyle = brushMode === "restore" ? "#ffffff" : "#000000";
  maskContext.lineCap = "round";
  maskContext.lineJoin = "round";
  maskContext.lineWidth = Number($("brush-size").value) * point.scale * 2;
  maskContext.beginPath();
  maskContext.arc(point.x, point.y, maskContext.lineWidth / 2, 0, Math.PI * 2);
  maskContext.fill();
  maskContext.beginPath();
  maskContext.moveTo(point.x, point.y);
}

function moveBrush(event) {
  if (!painting || !lastBrushPoint) return;
  const point = getBrushPoint(event);
  if (!point) return;
  event.preventDefault();
  maskContext.lineTo(point.x, point.y);
  maskContext.stroke();
  lastBrushPoint = point;
}

function endBrush() {
  if (!painting) return;
  painting = false;
  lastBrushPoint = null;
  maskContext.restore();
  const pixels = maskContext.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;
  maskData = new Uint8ClampedArray(maskCanvas.width * maskCanvas.height);
  for (let i = 0; i < maskData.length; i++) maskData[i] = pixels[i * 4];
  $("undo").disabled = false;
  renderOutput();
}

function undoBrush() {
  if (!undoData || !maskReady) return;
  maskContext.putImageData(undoData, 0, 0);
  const pixels = undoData.data;
  maskData = new Uint8ClampedArray(maskCanvas.width * maskCanvas.height);
  for (let i = 0; i < maskData.length; i++) maskData[i] = pixels[i * 4];
  undoData = null;
  $("undo").disabled = true;
  renderOutput();
}

function resetMask() {
  if (!baseMaskData) return;
  commitMask(baseMaskData, false);
  setStatus("Mask reset to the original segmentation result.");
}

function downloadPNG() {
  if (!maskReady || !resultCanvas.width) return;
  resultCanvas.toBlob((blob) => {
    if (!blob) {
      setStatus("PNG export failed. The browser could not encode this canvas.");
      return;
    }
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    const filename = (sourceFile && sourceFile.name) || "image";
    const basename = filename.lastIndexOf(".") > 0 ? filename.slice(0, filename.lastIndexOf(".")) : filename;
    link.download = basename + "-no-background.png";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setStatus("Transparent PNG exported at " + sourceWidth + " × " + sourceHeight + ".");
  }, "image/png");
}

$("choose-image").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (file) loadImage(file);
  fileInput.value = "";
});
dropzone.addEventListener("click", (event) => {
  if (event.target.closest("button")) return;
  fileInput.click();
});
for (const type of ["dragenter", "dragover"]) {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
}
for (const type of ["dragleave", "drop"]) {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
}
dropzone.addEventListener("drop", (event) => {
  const file = Array.from(event.dataTransfer.files || []).find((item) => item.type.startsWith("image/"));
  if (file) loadImage(file);
});
window.addEventListener("paste", (event) => {
  const items = Array.from(event.clipboardData && event.clipboardData.items || []);
  const item = items.find((entry) => entry.type.startsWith("image/"));
  if (item) {
    const file = item.getAsFile();
    if (file) loadImage(file);
  }
});

$("ai-remove").addEventListener("click", removeWithAI);
$("pick-background").addEventListener("click", toggleColorPick);
sourcePreview.addEventListener("click", sampleBackground);
resultCanvas.addEventListener("pointerdown", startBrush);
resultCanvas.addEventListener("pointermove", moveBrush);
resultCanvas.addEventListener("pointerup", endBrush);
resultCanvas.addEventListener("pointercancel", endBrush);
resultCanvas.addEventListener("lostpointercapture", endBrush);
$("brush-view").addEventListener("click", () => { brushMode = "none"; updateBrushButtons(); });
$("brush-restore").addEventListener("click", () => { brushMode = "restore"; updateBrushButtons(); });
$("brush-erase").addEventListener("click", () => { brushMode = "erase"; updateBrushButtons(); });
$("undo").addEventListener("click", undoBrush);
$("reset-mask").addEventListener("click", resetMask);
$("download").addEventListener("click", downloadPNG);
$("feather").addEventListener("input", () => { updateSliderLabels(); renderOutput(); });
for (const id of ["brush-size", "tolerance", "color-softness"]) {
  $(id).addEventListener("input", updateSliderLabels);
}
updateSliderLabels();
updateBrushButtons();
setMaskControls(false);
setProgress("Ready", 0, "Choose an image");
