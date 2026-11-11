import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";

const MODEL_BASE = "https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/resolve/main/";
const MANIFEST_URL = MODEL_BASE + "manifest.json";

ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

const button = document.querySelector("#testClipEncoder");
const status = document.querySelector("#clipSmokeStatus");
const results = document.querySelector("#clipSmokeSummary");

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

    setStatus("Fetching the small model manifest…");
    const manifest = await loadManifest();
    const component = manifest?.components?.text_encoder;
    if (!component?.onnx || !Array.isArray(component.shards) || !component.shards.length) {
      throw new Error("The model manifest is missing the text_encoder graph or its external-data shards.");
    }

    addRow("Model pack", manifest.packageId || "WAI Illustrious v17");
    addRow("Component", "CLIP-L text encoder");
    addRow("External weights", formatBytes(component.bytes));
    addRow("Execution provider", "WebGPU");
    addRow("Graph", component.onnx);

    const externalData = component.shards.map(path => ({
      // ONNX external-data locations are relative to the graph in its component
      // directory; the manifest stores the corresponding repository-relative URL.
      path: path.slice(path.lastIndexOf("/") + 1),
      data: MODEL_BASE + path
    }));

    setStatus("Loading CLIP-L into ONNX Runtime WebGPU. This downloads about 232 MB of weights; it does not download the UNet or the rest of the model.");
    const started = performance.now();
    session = await ort.InferenceSession.create(MODEL_BASE + component.onnx, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "basic",
      externalData
    });

    if (session.inputNames.length !== 1) {
      throw new Error("Expected one CLIP-L input, found: " + session.inputNames.join(", "));
    }

    setStatus("Running one smoke-test inference through the WebGPU session…");
    const input = new ort.Tensor("int64", new BigInt64Array(77), [1, 77]);
    const outputs = await session.run({ [session.inputNames[0]]: input });
    const outputName = session.outputNames[0];
    const output = outputs[outputName];

    if (!output || !output.data || !output.dims) {
      throw new Error("The CLIP-L session returned no readable output tensor.");
    }
    for (let i = 0; i < output.data.length; i++) {
      if (!Number.isFinite(Number(output.data[i]))) {
        throw new Error("CLIP-L returned a non-finite output at element " + i + ".");
      }
    }

    addRow("Session inputs", session.inputNames.join(", "));
    addRow("Session outputs", session.outputNames.join(", "));
    addRow("Test output", outputName);
    addRow("Output shape", output.dims.join(" × "));
    addRow("Output dtype", output.type);
    addRow("Output values", "All finite");
    addRow("Load + inference", ((performance.now() - started) / 1000).toFixed(1) + " s");

    setStatus("CLIP-L loaded and completed a test inference on WebGPU. This validates one model component only; it is not an image-generation test.", "success");
  } catch (error) {
    console.error("[Illustrious] CLIP-L smoke test failed:", error);
    setStatus("CLIP-L WebGPU smoke test failed: " + (error?.message || String(error)), "error");
  } finally {
    if (session) {
      try {
        await session.release();
      } catch (error) {
        console.warn("[Illustrious] Could not release CLIP-L session:", error);
      }
    }
    button.disabled = false;
  }
});
