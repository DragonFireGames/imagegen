import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";

const MODEL_BASE = "https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/resolve/main/";
const MANIFEST_URL = MODEL_BASE + "manifest.json";

ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

const button = document.querySelector("#testVaeDecoder");
const status = document.querySelector("#vaeSmokeStatus");
const results = document.querySelector("#vaeSmokeSummary");

function setStatus(message, type = "info") {
  status.textContent = message;
  status.dataset.type = type;
}

function addRow(label, value) {
  const row = document.createElement("div");
  row.className = "data-row";
  const labelNode = document.createElement("span");
  labelNode.className = "data-label";
  labelNode.textContent = label;
  const valueNode = document.createElement("span");
  valueNode.className = "data-value";
  valueNode.textContent = value;
  row.append(labelNode, valueNode);
  results.append(row);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "Unknown";
  if (bytes >= 1024 ** 3) return (bytes / 1024 ** 3).toFixed(2) + " GiB";
  return (bytes / 1024 ** 2).toFixed(1) + " MiB";
}

button.addEventListener("click", async () => {
  button.disabled = true;
  results.replaceChildren();
  results.hidden = false;

  let session;
  try {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("The browser did not return a WebGPU adapter.");

    setStatus("Fetching the model manifest…");
    const response = await fetch(MANIFEST_URL, { credentials: "omit" });
    if (!response.ok) throw new Error("Model manifest request failed: HTTP " + response.status);
    const manifest = await response.json();
    const component = manifest?.components?.vae_decoder;
    if (!component?.onnx || !Array.isArray(component.shards) || !component.shards.length) {
      throw new Error("The model manifest is missing the vae_decoder graph or its external-data shards.");
    }

    addRow("Model pack", manifest.packageId || "WAI Illustrious v17");
    addRow("Component", "VAE decoder");
    addRow("Component download", formatBytes(component.bytes));
    addRow("Execution provider", "WebGPU");
    addRow("Graph", component.onnx);

    const externalData = component.shards.map(path => ({
      path: path.slice(path.lastIndexOf("/") + 1),
      data: MODEL_BASE + path
    }));

    setStatus("Loading the VAE decoder and external weights…");
    const started = performance.now();
    session = await ort.InferenceSession.create(MODEL_BASE + component.onnx, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "basic",
      externalData
    });

    const inputName = session.inputNames[0];
    if (!inputName) throw new Error("The VAE decoder graph has no inputs.");
    const input = new ort.Tensor("float32", new Float32Array(4 * 128 * 128), [1, 4, 128, 128]);

    setStatus("Running a zero-latent decode through the WebGPU session…");
    const outputs = await session.run({ [inputName]: input });
    const outputName = session.outputNames[0];
    const output = outputs[outputName];
    if (!output?.data || !output?.dims) {
      throw new Error("The VAE decoder returned no readable output tensor.");
    }
    for (let i = 0; i < output.data.length; i++) {
      if (!Number.isFinite(Number(output.data[i]))) {
        throw new Error("VAE decoder returned a non-finite value at element " + i + ".");
      }
    }

    addRow("Session input", inputName + " · " + input.dims.join(" × "));
    addRow("Session output", outputName);
    addRow("Output shape", output.dims.join(" × "));
    addRow("Output dtype", output.type);
    addRow("Output values", "All finite");
    addRow("Load + inference", ((performance.now() - started) / 1000).toFixed(1) + " s");

    setStatus("VAE decoder loaded and completed a WebGPU test inference. This validates decoding only; it does not generate an image.", "success");
  } catch (error) {
    console.error("[Illustrious] VAE decoder smoke test failed:", error);
    setStatus("VAE decoder WebGPU smoke test failed: " + (error?.message || String(error)), "error");
  } finally {
    if (session) {
      try {
        await session.release();
      } catch (error) {
        console.warn("[Illustrious] Could not release VAE decoder session:", error);
      }
    }
    button.disabled = false;
  }
});
