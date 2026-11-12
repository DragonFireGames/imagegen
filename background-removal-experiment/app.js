import * as ort from "../../vendor/onnxruntime-web/ort.webgpu.min.mjs";

const MODEL_BASE = "https://huggingface.co/baby2008/RMBG-2.0-ONNX/resolve/78c06dfe901fc676de75ab1135ab34934dfb6666/onnx/";
const MODEL_FILES = { webgpu: "model_fp16.onnx", wasm: "model_int8.onnx" };
const MODEL_CACHE = "imagegen-rmbg-2-models-v1";
const MODEL_INPUT_SIZE = 1024;
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
const modelCanvas = document.createElement("canvas");
const modelContext = modelCanvas.getContext("2d", { willReadFrequently: true });
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
let sessions = {};

ort.env.wasm.wasmPaths = new URL("../../vendor/onnxruntime-web/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

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
  $("backend").disabled = busy;
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

function formatMiB(bytes) {
  return (bytes / 1024 / 1024).toFixed(0) + " MiB";
}

async function loadModelBytes(device) {
  const file = MODEL_FILES[device];
  const url = MODEL_BASE + file;
  let cache = null;

  try {
    if ("caches" in window) cache = await caches.open(MODEL_CACHE);
  } catch (error) {
    console.warn("Background Removal Lab: persistent model cache unavailable.", error);
  }

  if (cache) {
    try {
      const cached = await cache.match(url);
      if (cached) {
        setProgress("Loading cached RMBG-2.0 weights", 55, file);
        const buffer = await cached.arrayBuffer();
        setProgress("Cached model loaded", 78, formatMiB(buffer.byteLength));
        return buffer;
      }
    } catch (error) {
      console.warn("Background Removal Lab: cached model could not be read; downloading again.", error);
    }
  }

  setProgress("Downloading RMBG-2.0 weights", 5, file);
  const response = await fetch(url, { mode: "cors" });
  if (!response.ok) {
    throw new Error("Could not download " + file + " (" + response.status + "). Check the model host and network connection.");
  }

  let cacheWrite = Promise.resolve();
  if (cache) {
    cacheWrite = cache.put(url, response.clone()).catch((error) => {
      console.warn("Background Removal Lab: model could not be persisted in Cache Storage.", error);
    });
  }

  const total = Number(response.headers.get("content-length")) || 0;
  let loaded = 0;
  let buffer;

  if (response.body && typeof TransformStream !== "undefined") {
    const progressStream = response.body.pipeThrough(new TransformStream({
      transform(chunk, controller) {
        loaded += chunk.byteLength;
        const detail = total
          ? formatMiB(loaded) + " / " + formatMiB(total)
          : formatMiB(loaded) + " received";
        const percent = total ? 5 + Math.min(70, loaded / total * 70) : 8;
        setProgress("Downloading RMBG-2.0 weights", percent, detail);
        controller.enqueue(chunk);
      }
    }));
    buffer = await new Response(progressStream).arrayBuffer();
  } else {
    buffer = await response.arrayBuffer();
  }

  await cacheWrite;
  setProgress("RMBG-2.0 weights downloaded", 78, formatMiB(buffer.byteLength));
  return buffer;
}

async function getSession(device) {
  if (sessions[device]) return sessions[device];

  const file = MODEL_FILES[device];
  const precision = device === "webgpu" ? "FP16 weights" : "INT8 weights";
  setProgress("Preparing RMBG-2.0", 2, device.toUpperCase() + " · " + precision);
  setStatus("Loading RMBG-2.0 " + precision + " weights. The first run downloads the model; later runs use the browser cache.");

  const modelBytes = await loadModelBytes(device);
  setProgress("Initializing RMBG-2.0", 82, device.toUpperCase() + " · " + precision);

  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: [device],
    graphOptimizationLevel: "all"
  });

  const inputName = session.inputNames.find((name) => name === "pixel_values") || session.inputNames[0];
  const outputName = session.outputNames.find((name) => name === "alphas") || session.outputNames[0];
  if (!inputName || !outputName) {
    await session.release();
    throw new Error("Unexpected RMBG-2.0 ONNX graph inputs or outputs.");
  }

  sessions[device] = { session, inputName, outputName, file };
  return sessions[device];
}

function makeModelInput() {
  modelCanvas.width = MODEL_INPUT_SIZE;
  modelCanvas.height = MODEL_INPUT_SIZE;
  modelContext.clearRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  modelContext.imageSmoothingEnabled = true;
  modelContext.imageSmoothingQuality = "high";
  modelContext.drawImage(processCanvas, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

  const rgba = modelContext.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).data;
  const area = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
  const values = new Float32Array(area * 3);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let pixel = 0; pixel < area; pixel++) {
    const source = pixel * 4;
    values[pixel] = (rgba[source] / 255 - mean[0]) / std[0];
    values[area + pixel] = (rgba[source + 1] / 255 - mean[1]) / std[1];
    values[area * 2 + pixel] = (rgba[source + 2] / 255 - mean[2]) / std[2];
  }

  return new ort.Tensor("float32", values, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
}

function halfToFloat(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 31;
  const fraction = value & 1023;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function readMask(tensor) {
  if (!tensor || !tensor.data || !Array.isArray(tensor.dims) || tensor.dims.length < 2) {
    throw new Error("RMBG-2.0 did not return a readable alpha matte.");
  }

  const width = tensor.dims[tensor.dims.length - 1];
  const height = tensor.dims[tensor.dims.length - 2];
  const data = tensor.data;
  if (!width || !height || data.length < width * height) {
    throw new Error("RMBG-2.0 returned an unexpected alpha matte shape: " + tensor.dims.join(" × "));
  }

  const readValue = (index) => {
    let value = Number(data[index]);
    if (tensor.type === "float16" && data instanceof Uint16Array) value = halfToFloat(data[index]);
    if (!Number.isFinite(value)) return 0;
    if (value >= 0 && value <= 1) value *= 255;
    return Math.max(0, Math.min(255, value));
  };

  const output = new Uint8ClampedArray(processWidth * processHeight);
  for (let y = 0; y < processHeight; y++) {
    const sourceY = Math.max(0, Math.min(height - 1, (y + 0.5) * height / processHeight - 0.5));
    const y0 = Math.floor(sourceY);
    const y1 = Math.min(height - 1, y0 + 1);
    const fy = sourceY - y0;

    for (let x = 0; x < processWidth; x++) {
      const sourceX = Math.max(0, Math.min(width - 1, (x + 0.5) * width / processWidth - 0.5));
      const x0 = Math.floor(sourceX);
      const x1 = Math.min(width - 1, x0 + 1);
      const fx = sourceX - x0;

      const top = readValue(y0 * width + x0) * (1 - fx) + readValue(y0 * width + x1) * fx;
      const bottom = readValue(y1 * width + x0) * (1 - fx) + readValue(y1 * width + x1) * fx;
      output[y * processWidth + x] = Math.round(top * (1 - fy) + bottom * fy);
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
  $("model-status").textContent = "Loading RMBG-2.0";
  $("model-status").classList.remove("ready");

  const requested = $("backend").value;
  const candidates = requested === "auto"
    ? (navigator.gpu ? ["webgpu", "wasm"] : ["wasm"])
    : [requested];
  let lastError = null;

  try {
    if (navigator.storage && navigator.storage.persist) {
      navigator.storage.persist().catch(() => {});
    }

    setProgress("Preprocessing image", 1, "1024 × 1024");
    const inputTensor = makeModelInput();

    for (let i = 0; i < candidates.length; i++) {
      const device = candidates[i];
      try {
        if (device === "webgpu" && !navigator.gpu) {
          throw new Error("WebGPU is unavailable in this browser.");
        }

        const runtime = await getSession(device);
        setProgress("Segmenting foreground", 90, device.toUpperCase());
        setStatus("Running RMBG-2.0 locally on " + device.toUpperCase() + "…");

        const outputs = await runtime.session.run({ [runtime.inputName]: inputTensor });
        const alpha = outputs.alphas || outputs[runtime.outputName];
        const nextMask = readMask(alpha);
        let visible = 0;
        let minAlpha = 255;
        let maxAlpha = 0;
        for (let p = 0; p < nextMask.length; p++) {
          const value = nextMask[p];
          if (value > 4) visible++;
          if (value < minAlpha) minAlpha = value;
          if (value > maxAlpha) maxAlpha = value;
        }
        if (!visible || maxAlpha < 8 || maxAlpha - minAlpha < 1) {
          throw new Error("RMBG-2.0 returned an empty or constant alpha matte. The model output may be incompatible.");
        }

        commitMask(nextMask, true);
        const precision = device === "webgpu" ? "FP16 weights" : "INT8 weights";
        $("model-status").textContent = "RMBG-2.0 · " + device.toUpperCase();
        $("model-status").classList.add("ready");
        setProgress("Background removed", 100, device.toUpperCase() + " · " + precision);
        setStatus("RMBG-2.0 cutout complete. Restore and Erase refine the alpha matte.");
        $("source-tip").textContent = "RMBG-2.0 · 1024px model input";
        return;
      } catch (error) {
        lastError = error;
        if (device === "webgpu" && requested === "auto" && i + 1 < candidates.length) {
          const failed = sessions.webgpu;
          if (failed && failed.session) {
            try { await failed.session.release(); } catch (releaseError) { console.warn(releaseError); }
          }
          delete sessions.webgpu;
          setStatus("WebGPU failed; retrying with the quantized WebAssembly model. " + (error.message || error));
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
    setStatus("RMBG-2.0 failed: " + (error.message || error));
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
