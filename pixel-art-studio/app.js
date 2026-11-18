import { HF_REPO, modelFiles, loadModel } from "./engine.js";
import { ENCODER, encoderFiles, loadTextEncoder } from "./text.js";
const MODEL_REVISION = "0a29494e9fc020a26feb6f97ec2047fc207ce898";
const MODEL_BASE = "https://huggingface.co/" + HF_REPO + "/resolve/" + MODEL_REVISION;
const ENCODER_BASE = "https://huggingface.co/" + ENCODER.repo + "/resolve/" + ENCODER.revision;
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs";
const TOKENIZERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.min.mjs";
const CACHE_NAME = "pixel-foundry-model-v1";
const $ = (id) => document.getElementById(id);
let ort, model, encoder, backend = "wasm", controller = null, loading = false, loadingModel = false;
function setStatus(s) { $("status").textContent = s; }
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
async function fetchBytes(url, onProgress) {
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); const hit = await cache.match(url); if (hit) return new Uint8Array(await hit.arrayBuffer()); }
  catch (_) { cache = null; }
  const response = await fetch(url);
  if (!response.ok) throw new Error("Model download failed (" + response.status + "): " + url);
  const total = Number(response.headers.get("content-length")) || 0;
  let bytes;
  if (onProgress && response.body && total) {
    const reader = response.body.getReader(); bytes = new Uint8Array(total); let received = 0;
    for (;;) { const part = await reader.read(); if (part.done) break; if (received + part.value.length > total) throw new Error("Download larger than announced: " + url); bytes.set(part.value, received); received += part.value.length; onProgress(received, total); }
    if (received !== total) throw new Error("Download was incomplete: " + url);
  } else { bytes = new Uint8Array(await response.arrayBuffer()); if (onProgress) onProgress(bytes.length, total || bytes.length); }
  if (cache) try { await cache.put(url, new Response(bytes)); } catch (e) { console.warn("Could not cache model file:", e); }
  return bytes;
}
const downloadProgress = (label) => (got, total) => setProgress(total ? got / total : 0, label + ": " + (got / 1e6).toFixed(0) + " / " + (total ? (total / 1e6).toFixed(0) : "?") + " MB");
async function chooseBackend() {
  if (!navigator.gpu) return "wasm";
  try { return await navigator.gpu.requestAdapter() ? "webgpu" : "wasm"; } catch (_) { return "wasm"; }
}
async function loadLocalModel(ep) {
  return loadModel({ ort, files: modelFiles(MODEL_BASE), ep, fetchBytes, fixedCache: true, onProgress: downloadProgress("Downloading sprite model") });
}
async function boot() {
  if (model || loadingModel) return;
  loadingModel = true;
  clearError(); $("loadModel").disabled = true; $("loadModel").textContent = "Loading model…";
  try {
    setStatus("Loading browser inference runtime…"); $("phase").textContent = "Loading ONNX Runtime Web";
    ort = await import(ORT_URL);
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    backend = await chooseBackend();
    setStatus("Downloading local sprite model…"); model = await loadLocalModel(backend);
    setStatus(backend === "webgpu" ? "Warming up WebGPU…" : "Warming up CPU inference…");
    $("phase").textContent = backend === "webgpu" ? "Compiling GPU shaders" : "Preparing WebAssembly";
    const x = new Float32Array(model.d); model.writePixel(x, 0, model.BOS, 0, 0, 3);
    const cache = await model.cacheSet(1, 4); const a = await model.stepFixed(x, 1, 0, cache);
    model.constructor.release(cache); model.constructor.release(a.cache);
    setStatus("Local model ready · " + (backend === "webgpu" ? "WebGPU" : "CPU"));
    $("phase").textContent = backend === "webgpu" ? "Ready · GPU inference" : "Ready · CPU inference (slower)";
    setProgress(1); $("go").disabled = false; $("go").textContent = "✦ Generate sprites"; $("loadModel").textContent = "Model loaded ✓";
  } catch (error) {
    if (backend === "webgpu") {
      try {
        console.warn("WebGPU model load failed; falling back to CPU.", error);
        setStatus("WebGPU unavailable · switching to CPU…"); backend = "wasm"; model = await loadLocalModel("wasm");
        setStatus("Local model ready · CPU"); $("phase").textContent = "Ready · CPU inference (slower)";
        $("go").disabled = false; $("go").textContent = "✦ Generate sprites"; $("loadModel").textContent = "Model loaded ✓"; return;
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
  setStatus("Downloading text encoder to this browser…"); $("phase").textContent = "First-time text encoder download";
  const { Tokenizer } = await import(TOKENIZERS_URL);
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
  const settings = {
    n: count, size, seed, gShape: Number($("shape").value), gColour: Number($("colour").value),
    temperature: 1, topP: 1, chunk: 4, signal: controller.signal,
    onPixel: (ids, t, images, view) => setProgress((t + 1) / (view.S * view.S), "Drawing pixels · " + (t + 1) + " / " + (view.S * view.S)),
    onSprite: (id, image) => addSprite(image, size, id + 1)
  };
  try {
    setStatus("Encoding prompt locally…"); const enc = await loadEncoder(); settings.vector = await enc.encode(prompt);
    setStatus("Generating on your " + (backend === "webgpu" ? "GPU" : "CPU") + "…"); $("empty").classList.add("hidden");
    await model.generate(settings);
    setStatus("Finished · " + count + " sprites generated locally"); $("phase").textContent = "Complete"; setProgress(1);
  } catch (error) {
    if (error && error.name === "AbortError") { setStatus("Generation stopped"); $("phase").textContent = "Stopped"; }
    else if (backend === "webgpu" && model) {
      try {
        console.warn("GPU inference failed; retrying on CPU.", error); setStatus("GPU inference failed · switching to CPU…");
        try { await model.session.release(); } catch (_) {}
        backend = "wasm"; model = await loadLocalModel("wasm"); $("results").replaceChildren(); setStatus("Retrying generation on CPU…");
        await model.generate(settings); setStatus("Finished · generated locally on CPU"); $("phase").textContent = "Complete · CPU"; setProgress(1);
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
setStatus("Ready · model not loaded"); $("phase").textContent = "Load the model when you are ready";
