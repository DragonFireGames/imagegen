import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";

const MODEL_BASE = "https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/resolve/main/";
const MANIFEST_URL = MODEL_BASE + "manifest.json";

ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

const button = document.querySelector("#testClipBigG");
const status = document.querySelector("#clipBigGStatus");
const results = document.querySelector("#clipBigGSummary");

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

async function loadManifest() {
  const response = await fetch(MANIFEST_URL, { credentials: "omit" });
  if (!response.ok) throw new Error("Model manifest request failed: HTTP " + response.status);
  return response.json();
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
    const manifest = await loadManifest();
    const component = manifest?.components?.text_encoder_2;
    if (!component?.onnx || !Array.isArray(component.shards) || !component.shards.length) {
      throw new Error("The model manifest is missing the text_encoder_2 graph or its external-data shards.");
    }

    addRow("Model pack", manifest.packageId || "WAI Illustrious v17");
    addRow("Component", "CLIP-bigG text encoder");
    addRow("Component download", formatBytes(component.bytes));
    addRow("Execution provider", "WebGPU");
    addRow("Graph", component.onnx);

    const externalData = component.shards.map(path => ({
      path: path.slice(path.lastIndexOf("/") + 1),
      data: MODEL_BASE + path
    }));

    setStatus("Loading CLIP-bigG and its external weights. This component is about 1.39 GB; keep other memory-heavy tabs closed.");
    const started = performance.now();
    session = await ort.InferenceSession.create(MODEL_BASE + component.onnx, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "basic",
      externalData
    });

    if (session.inputNames.length !== 1) {
      throw new Error("Expected one CLIP-bigG input, found: " + session.inputNames.join(", "));
    }

    const inputName = session.inputNames[0];
    const inputMetadata = session.inputMetadata?.[0];
    if (inputMetadata?.type && inputMetadata.type !== "int64") {
      throw new Error("CLIP-bigG input " + inputName + " has unexpected type " + inputMetadata.type + "; expected int64.");
    }
    if (inputMetadata?.shape && inputMetadata.shape.length &&
        (inputMetadata.shape.length !== 2 ||
         (typeof inputMetadata.shape[0] === "number" && inputMetadata.shape[0] !== 1) ||
         (typeof inputMetadata.shape[1] === "number" && inputMetadata.shape[1] !== 77))) {
      throw new Error("CLIP-bigG input has unexpected shape " + inputMetadata.shape.join(" × ") + "; expected [1, 77].");
    }

    setStatus("Running one dummy-token inference through CLIP-bigG on WebGPU…");
    const input = new ort.Tensor("int64", new BigInt64Array(77), [1, 77]);
    const outputs = await session.run({ [inputName]: input });

    const outputNames = session.outputNames;
    if (!outputNames.length) throw new Error("CLIP-bigG graph exposes no outputs.");
    let totalValues = 0;
    for (const outputName of outputNames) {
      const output = outputs[outputName];
      if (!output || !output.data || !output.dims) {
        throw new Error("CLIP-bigG returned no readable tensor for output " + outputName + ".");
      }
      for (let i = 0; i < output.data.length; i++) {
        if (!Number.isFinite(Number(output.data[i]))) {
          throw new Error("CLIP-bigG output " + outputName + " contains a non-finite value at element " + i + ".");
        }
      }
      totalValues += output.data.length;
      addRow("Output " + outputName, output.dims.join(" × ") + " · " + output.type + " · all finite");
    }

    addRow("Session input", inputName + " · int64 · 1 × 77");
    addRow("Session outputs", outputNames.join(", "));
    addRow("Output values checked", totalValues.toLocaleString());
    addRow("Load + inference", ((performance.now() - started) / 1000).toFixed(1) + " s");

    setStatus("CLIP-bigG loaded and completed a test inference on WebGPU. All exposed outputs are finite. This validates the second text encoder only; it is not an image-generation test.", "success");
  } catch (error) {
    console.error("[Illustrious] CLIP-bigG smoke test failed:", error);
    setStatus("CLIP-bigG WebGPU smoke test failed: " + (error?.message || String(error)), "error");
  } finally {
    if (session) {
      try {
        await session.release();
      } catch (error) {
        console.warn("[Illustrious] Could not release CLIP-bigG session:", error);
      }
    }
    button.disabled = false;
  }
});
