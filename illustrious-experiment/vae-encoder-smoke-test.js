import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";
import { acquireHeavyModelTest, releaseHeavyModelTest } from "./heavy-model-test-lock.js";

const MODEL_BASE = "https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/resolve/main/";
const MANIFEST_URL = MODEL_BASE + "manifest.json";

ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

const button = document.querySelector("#testVaeEncoder");
const status = document.querySelector("#vaeEncoderStatus");
const results = document.querySelector("#vaeEncoderSummary");

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

function finiteFp16(tensor, label) {
  if (!tensor?.data || tensor.type !== "float16" || !tensor.dims) {
    throw new Error(label + " is not a readable FP16 tensor.");
  }
  if (!(tensor.data instanceof Float16Array) && !(tensor.data instanceof Uint16Array)) {
    throw new Error(label + " data is " + tensor.data.constructor.name +
      "; expected Float16Array or Uint16Array.");
  }
  const values = new Float16Array(tensor.data.buffer.slice(
    tensor.data.byteOffset,
    tensor.data.byteOffset + tensor.data.byteLength
  ));
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new Error(label + " contains a non-finite value at element " + i + ".");
    }
  }
  return values.length;
}

button.addEventListener("click", async () => {
  button.disabled = true;
  results.replaceChildren();
  results.hidden = false;

  let session;
  let lockAcquired = false;
  try {
    acquireHeavyModelTest("VAE encoder");
    lockAcquired = true;
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("The browser did not return a WebGPU adapter.");

    setStatus("Fetching the candidate model manifest…");
    const response = await fetch(MANIFEST_URL, { credentials: "omit" });
    if (!response.ok) throw new Error("Model manifest request failed: HTTP " + response.status);
    const manifest = await response.json();
    const component = manifest?.components?.vae_encoder;
    if (!component?.onnx || !Array.isArray(component.shards) || !component.shards.length) {
      throw new Error("The model manifest is missing the vae_encoder graph or its external-data shards.");
    }

    addRow("Model pack", manifest.packageId || "WAI Illustrious v17");
    addRow("Component", "VAE encoder");
    addRow("Component download", formatBytes(component.bytes));
    addRow("Execution provider", "WebGPU");
    addRow("Graph", component.onnx);

    const externalData = component.shards.map(path => ({
      path: path.slice(path.lastIndexOf("/") + 1),
      data: MODEL_BASE + path
    }));

    setStatus("Loading the VAE encoder and its external weights…");
    const started = performance.now();
    session = await ort.InferenceSession.create(MODEL_BASE + component.onnx, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "basic",
      externalData
    });

    if (session.inputNames.length !== 1) {
      throw new Error("Expected one VAE encoder input, found: " + session.inputNames.join(", "));
    }

    const inputName = session.inputNames[0];
    const metadata = session.inputMetadata?.[0];
    if (metadata?.type && metadata.type !== "float16") {
      throw new Error("VAE encoder input is " + metadata.type + "; expected float16.");
    }

    // Respect the graph's fixed dimensions when present. Symbolic spatial
    // dimensions are tested at 1024×1024, matching the SDXL decoder test.
    const declaredShape = metadata?.shape;
    let inputShape = [1, 3, 1024, 1024];
    if (Array.isArray(declaredShape) && declaredShape.length) {
      inputShape = declaredShape.map((dimension, index) => {
        if (typeof dimension === "number" && Number.isFinite(dimension) && dimension > 0) return dimension;
        return index === 0 ? 1 : index === 1 ? 3 : 1024;
      });
      if (inputShape.length !== 4 || inputShape[1] !== 3) {
        throw new Error("Unexpected VAE encoder input shape: [" + inputShape.join(", ") +
          "]; expected [1, 3, height, width].");
      }
    }

    const inputSize = inputShape.reduce((total, dimension) => total * dimension, 1);
    addRow("Session input", inputName + " · float16 · " + inputShape.join(" × "));
    addRow("Input values", inputSize.toLocaleString());
    addRow("Test image", "Zero-filled normalized RGB tensor (neutral gray in [-1, 1] convention).");

    const input = new ort.Tensor("float16", new Float16Array(inputSize), inputShape);
    setStatus("Running a zero-image encode through the VAE encoder on WebGPU…");
    const outputs = await session.run({ [inputName]: input });
    if (!session.outputNames.length) throw new Error("VAE encoder graph exposes no outputs.");

    let totalValues = 0;
    for (const name of session.outputNames) {
      const tensor = outputs[name];
      const count = finiteFp16(tensor, "VAE encoder output " + name);
      totalValues += count;
      addRow("Output " + name, tensor.dims.join(" × ") + " · " + tensor.type + " · all finite");
    }

    addRow("Output values checked", totalValues.toLocaleString());
    addRow("Load + inference", ((performance.now() - started) / 1000).toFixed(1) + " s");

    setStatus("VAE encoder loaded and completed a zero-image FP16 WebGPU inference. Output shapes are reported above; latent scaling and encoder/decoder round-trip compatibility are not yet assumed.", "success");
  } catch (error) {
    console.error("[Illustrious] VAE encoder smoke test failed:", error);
    setStatus("VAE encoder WebGPU smoke test failed: " + (error?.message || String(error)), "error");
  } finally {
    if (session) {
      try {
        await session.release();
      } catch (error) {
        console.warn("[Illustrious] Could not release VAE encoder session:", error);
      }
    }
    if (lockAcquired) releaseHeavyModelTest("VAE encoder");
    button.disabled = false;
  }
});
