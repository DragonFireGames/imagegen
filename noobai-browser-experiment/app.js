import { clearModelCache } from "./model-store.js";

const $ = (id) => document.getElementById(id);
const form = $("generation-form");
const workerUrl = new URL("./worker.js", import.meta.url);
let worker = null;
let runtimeReady = false;
let modelReady = false;
let busy = false;
let currentSeed = -1;

function logLine(value) {
  const log = $("log");
  const line = "[" + new Date().toLocaleTimeString() + "] " + value;
  log.textContent = (log.textContent ? log.textContent + "\n" : "") + line;
  log.scrollTop = log.scrollHeight;
}

function setProgress(label, percent = 0, extra = "") {
  $("progress-label").textContent = label;
  $("progress-extra").textContent = extra;
  $("progress-bar").style.width = Math.max(0, Math.min(100, percent)) + "%";
}

function setStatus(title, detail, type = "info") {
  $("setup-title").textContent = title;
  $("setup-detail").textContent = detail;
  $("runtime-pill").textContent = type === "ready" ? "Ready" : type === "error" ? "Unavailable" : "Setup";
  $("runtime-pill").dataset.type = type;
}

function setBusy(value) {
  busy = value;
  $("generate").disabled = value || !modelReady;
  $("generate").textContent = value ? "Generating…" : modelReady ? "Generate locally" : "Load the model pack first";
  $("load-model").disabled = value || !runtimeReady;
}

async function checkRuntime() {
  const issues = [];
  let adapter = null;
  if (!navigator.gpu) issues.push("WebGPU is unavailable.");
  else {
    try {
      adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) issues.push("No WebGPU adapter was returned.");
    } catch (error) {
      issues.push("WebGPU adapter check failed: " + error.message);
    }
  }
  if (!navigator.storage?.getDirectory) issues.push("OPFS is unavailable.");
  if (typeof WebAssembly.Suspending !== "function" || typeof WebAssembly.promising !== "function") {
    issues.push("JavaScript Promise Integration (JSPI) is unavailable. Use a recent Chromium browser with JSPI support.");
  }

  if (issues.length) {
    $("gpu-dot").classList.add("error");
    $("gpu-status").textContent = "Browser requirements missing";
    setStatus("This browser is not ready", issues.join(" "), "error");
    setProgress("Runtime requirements not met", 0);
    return;
  }

  $("gpu-dot").classList.add("ready");
  $("gpu-status").textContent = "WebGPU ready";
  runtimeReady = true;
  $("load-model").disabled = false;
  setStatus("Browser runtime ready", "WebGPU, OPFS and JSPI are available. Enter the published ONNX model-pack URL to download it into this browser.", "ready");
  setProgress("Ready for model pack", 0, adapter?.limits?.maxBufferSize ? (adapter.limits.maxBufferSize / 1024 ** 3).toFixed(1) + " GiB max GPU buffer" : "");
}

async function updateStorageStatus() {
  try {
    const estimate = await navigator.storage.estimate();
    const persisted = await navigator.storage.persisted?.();
    const used = Number(estimate.usage || 0) / 1024 ** 3;
    const quota = Number(estimate.quota || 0) / 1024 ** 3;
    $("storage-status").textContent = (persisted ? "Persistent · " : "Best-effort · ") +
      used.toFixed(1) + " / " + quota.toFixed(1) + " GB";
  } catch {
    $("storage-status").textContent = "Unavailable";
  }
}

function ensureWorker() {
  if (worker) return worker;
  worker = new Worker(workerUrl, { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.type === "status") {
      setProgress(data.label, 0, data.extra || "");
      logLine(data.label + (data.extra ? " — " + data.extra : ""));
    } else if (data.type === "progress") {
      setProgress(data.label, data.percent, data.total ? (data.loaded / 1024 ** 3).toFixed(2) + " / " + (data.total / 1024 ** 3).toFixed(2) + " GB" : "");
    } else if (data.type === "model-ready") {
      modelReady = true;
      $("model-status").textContent = "Cached · " + (data.totalBytes / 1024 ** 3).toFixed(2) + " GB";
      $("load-model").disabled = false;
      $("load-model").textContent = "Model cached — reload model";
      $("generate").disabled = false;
      $("generate").textContent = "Generate locally";
      setProgress("Model ready", 100, "Weights saved in OPFS");
      setStatus("NoobAI XL 1.1 is cached", "The model files are stored locally. Generation runs in this browser; no inference server is involved.", "ready");
      logLine("Model pack loaded and cached.");
      updateStorageStatus();
    } else if (data.type === "step") {
      setProgress("Denoising image", 20 + data.step / data.total * 65, "Step " + data.step + " / " + data.total);
    } else if (data.type === "result") {
      try {
        renderImage(data.buffer, data.dims);
        currentSeed = data.seed;
        $("seed").value = data.seed;
        $("seed-info").textContent = "Seed " + data.seed;
        $("result-info").textContent = Number(data.elapsed).toFixed(1) + "s · " + $("width").value + " × " + $("height").value + " · " + (data.sampler === "euler_a" ? "Euler a" : "Euler");
        setProgress("Generation complete", 100, "Seed " + data.seed);
        logLine("Image generated locally in " + Number(data.elapsed).toFixed(1) + " seconds.");
      } catch (error) {
        $("result-info").textContent = error.message;
        logLine("ERROR: " + error.message);
      }
      setBusy(false);
    } else if (data.type === "error") {
      logLine("ERROR: " + data.message);
      $("result-info").textContent = data.message.split("\n")[0];
      setProgress("Operation failed", 0);
      if (!modelReady) {
        setStatus("Model pack could not be loaded", data.message.split("\n")[0], "error");
        $("model-status").textContent = "Not loaded";
      }
      setBusy(false);
    } else if (data.type === "cache-cleared") {
      modelReady = false;
      $("model-status").textContent = "Cache cleared";
      $("load-model").textContent = "Load and cache model";
      $("generate").disabled = true;
      $("generate").textContent = "Load the model pack first";
      setProgress("Browser model cache cleared", 0);
      logLine("The NoobAI ONNX model cache was removed from OPFS.");
      updateStorageStatus();
    }
  };
  worker.onerror = (event) => {
    logLine("Worker error: " + (event.message || "Unknown worker error"));
    $("result-info").textContent = event.message || "Inference worker failed to start.";
    setProgress("Worker failed", 0);
    setBusy(false);
  };
  worker.onmessageerror = () => {
    logLine("Worker response could not be deserialized.");
    setBusy(false);
  };
  return worker;
}

function renderImage(buffer, dims) {
  if (!Array.isArray(dims) || dims.length !== 4 || dims[0] !== 1 || dims[1] !== 3) {
    throw new Error("Unexpected VAE output shape: " + (dims || []).join(" × "));
  }
  const height = dims[2];
  const width = dims[3];
  const data = new Float32Array(buffer);
  const plane = width * height;
  if (data.length !== plane * 3) throw new Error("Image output buffer does not match the declared dimensions.");
  const canvas = $("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { alpha: false });
  const image = context.createImageData(width, height);
  for (let pixel = 0; pixel < plane; pixel++) {
    const out = pixel * 4;
    image.data[out] = Math.max(0, Math.min(255, Math.round((data[pixel] * .5 + .5) * 255)));
    image.data[out + 1] = Math.max(0, Math.min(255, Math.round((data[plane + pixel] * .5 + .5) * 255)));
    image.data[out + 2] = Math.max(0, Math.min(255, Math.round((data[plane * 2 + pixel] * .5 + .5) * 255)));
    image.data[out + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  canvas.hidden = false;
  $("placeholder").hidden = true;
  $("download").hidden = false;
}

$("cfg").addEventListener("input", () => {
  $("cfg-value").textContent = Number($("cfg").value).toFixed(1);
});

for (const preset of document.querySelectorAll(".preset")) {
  preset.addEventListener("click", () => {
    $("width").value = preset.dataset.width;
    $("height").value = preset.dataset.height;
    document.querySelectorAll(".preset").forEach((button) => button.classList.toggle("active", button === preset));
  });
}

for (const field of [$("width"), $("height")]) {
  field.addEventListener("change", () => {
    document.querySelectorAll(".preset").forEach((button) => {
      button.classList.toggle("active",
        Number(button.dataset.width) === Number($("width").value) &&
        Number(button.dataset.height) === Number($("height").value));
    });
  });
}

async function loadModel(baseUrl = $("model-url").value.trim().replace(/\/+$/, "")) {
  if (!baseUrl.startsWith("https://")) {
    setStatus("Enter a public HTTPS model-pack URL", "The URL should point to the uploaded browser-pack folder, ending in /resolve/main/browser-pack.", "error");
    $("model-url").focus();
    return;
  }
  localStorage.setItem("noobai-browser-model-url", baseUrl);
  modelReady = false;
  $("model-status").textContent = "Downloading / checking cache…";
  $("load-model").disabled = true;
  $("generate").disabled = true;
  setProgress("Reading model manifest", 0);
  try {
    if (navigator.storage.persist) {
      const persisted = await navigator.storage.persist();
      logLine("Persistent storage request: " + (persisted ? "granted" : "not granted; browser may evict cached data"));
    }
    ensureWorker().postMessage({ op: "load", baseUrl });
  } catch (error) {
    logLine(error.message);
    setStatus("Could not start model loading", error.message, "error");
    $("load-model").disabled = false;
  }
}

$("load-model").addEventListener("click", () => loadModel());

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!modelReady || busy) return;
  const prompt = $("prompt").value.trim();
  if (!prompt) {
    $("prompt").focus();
    return;
  }
  const width = Number($("width").value);
  const height = Number($("height").value);
  if (![width, height].every((value) => Number.isInteger(value) && value >= 512 && value <= 1216 && value % 64 === 0)) {
    $("result-info").textContent = "Dimensions must be 512–1216 and divisible by 64.";
    return;
  }
  const steps = Number($("steps").value);
  const cfgScale = Number($("cfg").value);
  const requestedSeed = Number($("seed").value);
  if (!Number.isInteger(steps) || steps < 10 || steps > 50 || !Number.isFinite(cfgScale) || cfgScale < 1 || cfgScale > 12 ||
      !Number.isInteger(requestedSeed) || requestedSeed < -1 || requestedSeed > 2147483647) {
    $("result-info").textContent = "Check steps, CFG and seed values.";
    return;
  }

  setBusy(true);
  $("download").hidden = true;
  $("canvas").hidden = true;
  $("placeholder").hidden = false;
  $("placeholder").querySelector("h3").textContent = "Generating locally…";
  $("placeholder").querySelector("p").textContent = "The text encoders, U-Net and VAE run in this browser. No prompt is uploaded.";
  $("seed-info").textContent = "";
  $("result-info").textContent = "Preparing browser inference";
  setProgress("Starting inference", 2);
  ensureWorker().postMessage({
    op: "generate",
    prompt,
    negativePrompt: $("negative-prompt").value,
    width,
    height,
    steps,
    cfgScale,
    requestedSeed,
    sampler: $("sampler").value,
  });
});

$("download").addEventListener("click", () => {
  $("canvas").toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "noobai-xl-1.1-" + currentSeed + ".png";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, "image/png");
});

$("clear-cache").addEventListener("click", async () => {
  if (!confirm("Remove the cached NoobAI model files from this browser? The next load will download them again.")) return;
  localStorage.removeItem("noobai-browser-model-url");
  try {
    if (worker) {
      worker.postMessage({ op: "clear-cache" });
    } else {
      await clearModelCache();
      modelReady = false;
      $("model-status").textContent = "Cache cleared";
      $("load-model").textContent = "Load and cache model";
      $("generate").disabled = true;
      $("generate").textContent = "Load the model pack first";
      setProgress("Browser model cache cleared", 0);
      updateStorageStatus();
    }
  } catch (error) {
    logLine("Could not clear cache: " + error.message);
  }
});

$("model-url").value = localStorage.getItem("noobai-browser-model-url") || "";
(async () => {
  await updateStorageStatus();
  await checkRuntime();
  if (runtimeReady && $("model-url").value.trim()) {
    logLine("A previously selected model pack was found; checking its local cache.");
    await loadModel();
  }
})();
