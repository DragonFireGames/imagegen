import { Flux2Pipeline } from "./pipeline.js";

const $ = (id) => document.getElementById(id);
const generateButton = $("generate");
const canvas = $("canvas");
const placeholder = $("placeholder");
const progressBar = $("progress-bar");
const progressLabel = $("progress-label");
const progressExtra = $("progress-extra");
const downloadButton = $("download");
let pipeline;
let busy = false;

function setProgress(label, percent = 0, extra = "") {
  progressLabel.textContent = label;
  progressLabel.classList.remove("error");
  progressBar.style.width = Math.max(0, Math.min(100, percent)) + "%";
  progressExtra.textContent = extra;
}

function showError(error) {
  const message = error?.message || String(error);
  progressLabel.textContent = "Generation failed";
  progressLabel.classList.add("error");
  progressBar.style.width = "0%";
  progressExtra.textContent = "";
  $("result-info").textContent = message;
  placeholder.hidden = false;
  placeholder.querySelector("h3").textContent = "Generation failed";
  placeholder.querySelector("p").textContent = message.slice(0, 500);
}

function updateProgress(message) {
  if (message.type === "step") {
    setProgress("Denoising image", 18 + (message.step / message.total) * 62, "Step " + message.step + " / " + message.total);
    return;
  }
  const text = message.text || "Working";
  let percent = 5;
  if (text.includes("text encoder") || text.includes("Tokenizer")) percent = 8;
  else if (text.includes("transformer")) percent = 18;
  else if (text.includes("VAE")) percent = 84;
  setProgress(text, percent, "");
}

function renderImage(buffer, dims) {
  if (!Array.isArray(dims) || dims.length !== 4 || dims[0] !== 1 || dims[1] !== 3) {
    throw new Error("Unexpected VAE output shape: " + (dims || []).join(" × "));
  }
  const height = dims[2];
  const width = dims[3];
  const data = new Float32Array(buffer);
  if (data.length !== height * width * 3) throw new Error("VAE output buffer size does not match its dimensions.");

  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  const image = context.createImageData(width, height);
  const plane = width * height;
  for (let pixel = 0; pixel < plane; pixel++) {
    const offset = pixel * 4;
    image.data[offset] = Math.max(0, Math.min(255, Math.round(data[pixel] * 255)));
    image.data[offset + 1] = Math.max(0, Math.min(255, Math.round(data[plane + pixel] * 255)));
    image.data[offset + 2] = Math.max(0, Math.min(255, Math.round(data[plane * 2 + pixel] * 255)));
    image.data[offset + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  canvas.hidden = false;
  placeholder.hidden = true;
  downloadButton.hidden = false;
}

function chosenSeed() {
  const value = Number($("seed").value);
  return Number.isFinite(value) && value >= 0 ? Math.floor(value) : Math.floor(Math.random() * 2147483647);
}

async function checkWebGPU() {
  if (!navigator.gpu) {
    $("gpu-status").textContent = "WebGPU unavailable";
    setProgress("WebGPU is not available in this browser", 0, "Try a current Chrome or Edge release");
    $("result-info").textContent = "A WebGPU-capable browser is required.";
    generateButton.textContent = "WebGPU unavailable";
    return;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("The browser did not return a WebGPU adapter.");
    const maxBufferSize = adapter.limits?.maxBufferSize || 0;
    const extra = maxBufferSize ? (maxBufferSize / 1024 ** 3).toFixed(1) + " GiB max buffer" : "WebGPU adapter ready";
    $("gpu-dot").classList.add("ready");
    $("gpu-status").textContent = "WebGPU ready";
    generateButton.disabled = false;
    generateButton.textContent = "Generate image";
    setProgress("Ready", 0, extra);
  } catch (error) {
    $("gpu-status").textContent = "WebGPU adapter unavailable";
    setProgress("Could not access a WebGPU adapter", 0, error.message);
    generateButton.textContent = "WebGPU unavailable";
  }
}

async function generate() {
  if (busy) return;
  const prompt = $("prompt").value.trim();
  if (!prompt) {
    $("prompt").focus();
    setProgress("Enter a prompt first");
    return;
  }

  busy = true;
  generateButton.disabled = true;
  generateButton.textContent = "Generating…";
  downloadButton.hidden = true;
  $("seed-info").textContent = "";
  $("result-info").textContent = "Preparing local inference";
  const width = Number($("resolution").value);
  const height = width;
  const steps = Number($("steps").value);
  const seed = chosenSeed();

  try {
    pipeline ||= new Flux2Pipeline(updateProgress);
    const result = await pipeline.generate({ prompt, width, height, steps, seed });
    renderImage(result.buffer, result.dims);
    $("seed-info").textContent = "Seed " + seed;
    $("result-info").textContent = width + " × " + height + " · " + steps + " steps · " + result.elapsed.toFixed(1) + "s";
    setProgress("Complete", 100, "Seed " + seed);
  } catch (error) {
    console.error(error);
    showError(error);
  } finally {
    busy = false;
    generateButton.disabled = false;
    generateButton.textContent = "Generate image";
  }
}

generateButton.addEventListener("click", generate);
downloadButton.addEventListener("click", () => {
  canvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "imagegen-" + Date.now() + ".png";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, "image/png");
});

checkWebGPU();
