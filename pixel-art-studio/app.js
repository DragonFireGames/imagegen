import { HF_REPO, modelFiles, loadModel } from "./engine.js";
import { ENCODER, encoderFiles, loadTextEncoder } from "./text.js";
const MODEL_REVISION = "0a29494e9fc020a26feb6f97ec2047fc207ce898";
const MODEL_BASE = "https://huggingface.co/" + HF_REPO + "/resolve/" + MODEL_REVISION;
const ENCODER_BASE = "https://huggingface.co/" + ENCODER.repo + "/resolve/" + ENCODER.revision;
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs";
const TOKENIZERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.min.mjs";
const CACHE_NAME = "pixel-foundry-model-v1";
const $ = (id) => document.getElementById(id);
let ort, model, encoder, backend = "wasm", controller = null, loading = false, loadingModel = false, inferenceStarted = 0;
function setStatus(s) { $("status").textContent = s; }
function setGenerationComplete(count, seed) {
  const seconds = Math.max((performance.now() - inferenceStarted) / 1000, 0.001);
  setStatus("Finished · " + count + " sprites · seed " + seed + " · " + seconds.toFixed(1) + " s · " + (count / seconds).toFixed(2) + " sprites/s");
}
function setProgress(value, label) {
  $("progress").style.width = Math.max(0, Math.min(100, value * 100)) + "%";
  $("percent").textContent = value ? Math.round(value * 100) + "%" : "";
  if (label) $("phase").textContent = label;
}
function showError(error) {
  $("error").textContent = error && (error.stack || error.message) || String(error);
  $("error").classList.remove("hidden");
  console.error(error);
}
function clearError() { $("error").textContent = ""; $("error").classList.add("hidden"); }
function inferenceChunk(rowsPerSprite = 2) {
  const memory = Number(navigator.deviceMemory);
  const cores = navigator.hardwareConcurrency || 4;
  const lowMemory = Number.isFinite(memory) && memory > 0 ? memory <= 4 : cores <= 4;
  // Auto keeps the network-row budget (and therefore KV-cache memory) conservative.
  // Fast mode uses one conditional row per sprite; guided mode uses conditional + unconditional.
  const rowBudget = backend === "webgpu" && memory >= 8 && cores >= 8 ? 12 : lowMemory ? 4 : 8;
  const automatic = Math.max(1, Math.floor(rowBudget / rowsPerSprite));
  const selected = $("batch")?.value || "auto";
  return selected === "auto" ? automatic : Math.max(1, Math.min(8, Number(selected) || automatic));
}
async function warmupDecoder() {
  const requestedSize = Number($("size").value) || 32;
  const size = backend === "webgpu" ? (requestedSize <= 16 ? 16 : requestedSize === 24 ? 32 : requestedSize) : 16;
  const rowsPerSprite = $("mode").value === "fast" ? 1 : 2;
  const batch = backend === "webgpu" ? Math.min(inferenceChunk(rowsPerSprite), Number($("count").value) || 4) * rowsPerSprite : 1;
  const cacheLength = backend === "webgpu" ? 2 + size + size * size : 4;
  const x = new Float32Array(batch * model.d);
  for (let row = 0; row < batch; row++) model.writePixel(x, row * model.d, model.BOS, 0, 0, 3);
  const cache = await model.cacheSet(batch, cacheLength);
  let output = null;
  try {
    output = await model.stepFixed(x, batch, 0, cache);
  }
  finally {
    model.constructor.release(cache);
    if (output) model.constructor.release(output.cache);
  }
}
async function prepareOfflineSupport() {
  if (!("serviceWorker" in navigator)) throw new Error("Offline model caching requires a browser with service-worker support. Open this page over HTTPS in a supported browser.");
  await navigator.serviceWorker.register("./sw.js", { scope: "./", updateViaCache: "none" });
  await navigator.serviceWorker.ready;
  if (!navigator.serviceWorker.controller) {
    await new Promise((resolve, reject) => {
      const finish = () => { clearTimeout(timeout); resolve(); };
      const timeout = setTimeout(() => { navigator.serviceWorker.removeEventListener("controllerchange", finish); reject(new Error("Offline cache is not controlling this page yet. Reload the page and press Load model again.")); }, 10000);
      navigator.serviceWorker.addEventListener("controllerchange", finish, { once: true });
      if (navigator.serviceWorker.controller) finish();
    });
  }
}
async function fetchBytes(url, onProgress) {
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); const hit = await cache.match(url); if (hit) return new Uint8Array(await hit.arrayBuffer()); }
  catch (_) { cache = null; }
  const response = await fetch(url);
  if (!response.ok) throw new Error("Model download failed (" + response.status + "): " + url);
  const total = Number(response.headers.get("content-length")) || 0;
  let bytes;
  if (onProgress && response.body) {
    const reader = response.body.getReader(); let received = 0;
    if (total) {
      bytes = new Uint8Array(total);
      for (;;) { const part = await reader.read(); if (part.done) break; if (received + part.value.length > total) throw new Error("Download larger than announced: " + url); bytes.set(part.value, received); received += part.value.length; onProgress(received, total); }
      if (received !== total) throw new Error("Download was incomplete: " + url);
    } else {
      const parts = [];
      for (;;) { const part = await reader.read(); if (part.done) break; parts.push(part.value); received += part.value.length; onProgress(received, 0); }
      bytes = new Uint8Array(received); let offset = 0;
      for (const part of parts) { bytes.set(part, offset); offset += part.length; }
    }
  } else { bytes = new Uint8Array(await response.arrayBuffer()); if (onProgress) onProgress(bytes.length, total || bytes.length); }
  if (!cache) throw new Error("Browser storage is unavailable; model files cannot be saved for offline use.");
  try { await cache.put(url, new Response(bytes)); }
  catch (e) { throw new Error("Downloaded a model file but could not save it in browser storage. Check available site storage and try again. " + e.message); }
  return bytes;
}
const downloadProgress = (label) => (got, total) => setProgress(total ? got / total : 0, label + ": " + (got / 1e6).toFixed(0) + " / " + (total ? (total / 1e6).toFixed(0) : "?") + " MB");
async function chooseBackend() {
  if (!navigator.gpu) return "wasm";
  try {
    if (await navigator.gpu.requestAdapter({ powerPreference: "high-performance" })) {
      ort.env.webgpu.powerPreference = "high-performance"; return "webgpu";
    }
    if (await navigator.gpu.requestAdapter({ powerPreference: "low-power" })) {
      ort.env.webgpu.powerPreference = "low-power"; return "webgpu";
    }
  } catch (_) {}
  return "wasm";
}
async function loadLocalModel(ep) {
  return loadModel({ ort, files: modelFiles(MODEL_BASE), ep, fetchBytes, fixedCache: true, onProgress: downloadProgress("Downloading sprite model") });
}
async function boot() {
  if (model || loadingModel) return;
  loadingModel = true;
  clearError(); $("loadModel").disabled = true; $("loadModel").textContent = "Preparing offline cache…";
  try {
    setStatus("Preparing offline cache…"); $("phase").textContent = "Setting up browser storage";
    await prepareOfflineSupport();
    setStatus("Loading browser inference runtime…"); $("phase").textContent = "Loading ONNX Runtime Web";
    ort = await import(ORT_URL);
    ort.env.wasm.numThreads = self.crossOriginIsolated ? 0 : 1;
    backend = await chooseBackend();
    // ONNX Runtime WASM environment flags must be set before the first session.
    // Keep this fixed: WebGPU sessions may initialize WASM for CPU-assigned ops.
    ort.env.wasm.proxy = backend === "wasm";
    setStatus("Downloading local sprite model…"); model = await loadLocalModel(backend);
    setStatus(backend === "webgpu" ? "Warming up WebGPU…" : "Warming up CPU inference…");
    $("phase").textContent = "Preparing the normal batch size and 32 × 32 cache";
    await warmupDecoder();
    setStatus("Loading text encoder for offline use…"); $("phase").textContent = "Caching and warming the text encoder";
    await loadEncoder();
    setStatus("Local model ready · " + (backend === "webgpu" ? "WebGPU" : "CPU") + " · offline");
    $("phase").textContent = backend === "webgpu" ? "Ready · GPU inference" : "Ready · CPU inference (slower)";
    setProgress(1); $("go").disabled = false; $("go").textContent = "✦ Generate sprites"; $("loadModel").textContent = "Model loaded · offline ✓";
  } catch (error) {
    if (backend === "webgpu") {
      try {
        console.warn("WebGPU model load failed; falling back to CPU.", error);
        try { if (model && model.session) await model.session.release(); } catch (_) {}
        model = null; setStatus("WebGPU unavailable · switching to CPU…"); backend = "wasm"; model = await loadLocalModel("wasm");
        setStatus("Warming up CPU inference…"); $("phase").textContent = "Preparing the normal batch size and 32 × 32 cache"; await warmupDecoder();
        setStatus("Loading text encoder for offline use…"); $("phase").textContent = "Caching and warming the text encoder"; await loadEncoder();
        setStatus("Local model ready · CPU · offline"); $("phase").textContent = "Ready · CPU inference (slower)";
        $("go").disabled = false; $("go").textContent = "✦ Generate sprites"; $("loadModel").textContent = "Model loaded · offline ✓"; return;
      } catch (fallbackError) { model = null; showError(fallbackError); }
    } else { model = null; showError(error); }
    setStatus("Could not load the local model"); $("phase").textContent = "See error details"; $("go").disabled = true;
  } finally {
    loadingModel = false;
    $("loadModel").disabled = !!model;
    if (!model) $("loadModel").textContent = "↧ Load model";
  }
}
async function loadEncoder() {
  if (encoder) return encoder;
  setStatus("Loading text tokenizer…"); setProgress(0, "Loading text tokenizer…");
  const { Tokenizer } = await import(TOKENIZERS_URL);
  setStatus("Preparing text encoder files…"); setProgress(0, "Preparing text encoder files…");
  encoder = await loadTextEncoder({ ort, Tokenizer, files: encoderFiles(ENCODER_BASE), fetchBytes, onProgress: downloadProgress("Downloading text encoder") });
  return encoder;
}
function addSprite(indices, size, index) {
  const canvas = document.createElement("canvas"); canvas.width = size; canvas.height = size; canvas.className = "sprite";
  const ctx = canvas.getContext("2d"); const pixels = ctx.createImageData(size, size);
  pixels.data.set(model.rgba(indices)); ctx.putImageData(pixels, 0, 0);
  const tile = document.createElement("div"); tile.className = "tile"; tile.appendChild(canvas);
  const link = document.createElement("a"); link.textContent = "Download PNG ↓"; link.href = canvas.toDataURL("image/png"); link.download = "pixel-sprite-" + index + ".png";
  tile.appendChild(link); $("empty").classList.add("hidden"); $("results").appendChild(tile);
}
async function generate() {
  if (loading || !model) return;
  const prompt = $("prompt").value.trim(); if (!prompt) { $("prompt").focus(); return; }
  loading = true; clearError(); setProgress(0, "Preparing prompt…"); $("go").disabled = false; $("go").textContent = "Stop generation";
  const empty = $("empty"); $("results").replaceChildren(empty); $("caption").textContent = prompt; empty.classList.remove("hidden");
  $("empty").textContent = "Preparing prompt and generating sprites locally…"; controller = new AbortController();
  const size = Number($("size").value), count = Number($("count").value);
  let seed = Number($("seed").value); if (!Number.isFinite(seed)) seed = 0;
  const fastMode = $("mode").value === "fast";
  const rowsPerSprite = fastMode ? 1 : 2;
  const settings = {
    n: count, size, seed, gShape: fastMode ? 1 : Number($("shape").value), gColour: fastMode ? 1 : Number($("colour").value),
    filter: !fastMode, temperature: 1, topP: 1, chunk: inferenceChunk(rowsPerSprite), signal: controller.signal,
    onPixel: (ids, t, images, view) => {
      if ((t + 1) % view.S === 0 || t + 1 === view.S * view.S)
        setProgress((t + 1) / (view.S * view.S), "Drawing row · " + Math.ceil((t + 1) / view.S) + " / " + view.S);
    },
    onSprite: (id, image) => addSprite(image, size, id + 1)
  };
  try {
    setStatus("Encoding prompt locally…"); const enc = await loadEncoder(); settings.vector = await enc.encode(prompt);
    setStatus("Generating on your " + (backend === "webgpu" ? "GPU" : "CPU") + "…"); $("empty").classList.add("hidden");
    inferenceStarted = performance.now();
    const result = await model.generate(settings);
    setGenerationComplete(count, result.stats.seed); $("phase").textContent = "Complete"; setProgress(1);
  } catch (error) {
    if (error && error.name === "AbortError") { setStatus("Generation stopped"); $("phase").textContent = "Stopped"; }
    else if (backend === "webgpu" && model) {
      try {
        console.warn("GPU inference failed; retrying on CPU.", error); setStatus("GPU inference failed · switching to CPU…");
        try { await model.session.release(); } catch (_) {}
        backend = "wasm"; settings.chunk = inferenceChunk(rowsPerSprite); model = await loadLocalModel("wasm"); $("results").replaceChildren();
        setStatus("Warming up CPU inference…"); await warmupDecoder(); setStatus("Retrying generation on CPU…");
        inferenceStarted = performance.now();
        const result = await model.generate(settings); setGenerationComplete(count, result.stats.seed); $("phase").textContent = "Complete · CPU"; setProgress(1);
      } catch (fallbackError) { showError(fallbackError); setStatus("Generation failed"); }
    } else { showError(error); setStatus("Generation failed"); $("phase").textContent = "See error details"; }
  } finally { loading = false; controller = null; $("go").disabled = false; $("go").textContent = "✦ Generate sprites"; }
}
$("examples").addEventListener("click", (event) => { const button = event.target.closest("[data-p]"); if (button) $("prompt").value = button.dataset.p; });
$("randomSeed").addEventListener("click", () => $("seed").value = Math.floor(Math.random() * 2147483647));
$("shape").addEventListener("input", () => $("shapeValue").textContent = Number($("shape").value).toFixed(2));
$("colour").addEventListener("input", () => $("colourValue").textContent = Number($("colour").value).toFixed(2));
$("go").addEventListener("click", () => { if (loading && controller) { controller.abort(); $("go").disabled = true; $("go").textContent = "Stopping…"; } else generate(); });
$("prompt").addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") generate(); });
$("clear").addEventListener("click", () => { $("results").replaceChildren($("empty")); $("empty").classList.remove("hidden"); $("empty").textContent = "No sprites yet. Write a prompt and generate a batch."; $("caption").textContent = "Sprites will appear here."; });
$("loadModel").addEventListener("click", boot);
setStatus("Ready · model not loaded"); $("phase").textContent = "Load and cache everything for offline inference";
