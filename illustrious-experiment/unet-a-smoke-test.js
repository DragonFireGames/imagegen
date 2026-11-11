import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";

const MODEL_BASE = "https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/resolve/main/";
const MANIFEST_URL = MODEL_BASE + "manifest.json";

ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

const button = document.querySelector("#testUnetA");
const status = document.querySelector("#unetAStatus");
const results = document.querySelector("#unetASummary");

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

    setStatus("Fetching the manifest for UNET_A…");
    const manifest = await loadManifest();
    const component = manifest?.components?.unet_a;
    if (!component?.onnx || !Array.isArray(component.shards) || !component.shards.length) {
      throw new Error("The model manifest is missing the unet_a graph or its external-data shards.");
    }

    addRow("Model pack", manifest.packageId || "WAI Illustrious v17");
    addRow("Component", "UNET_A · down path + middle");
    addRow("Component download", formatBytes(component.bytes));
    addRow("External-data shards", String(component.shards.length));
    addRow("Execution provider", "WebGPU");
    addRow("Graph", component.onnx);
    addRow("Input resolution", "1024 × 1024 · latent 128 × 128");

    const externalData = component.shards.map(path => ({
      path: path.slice(path.lastIndexOf("/") + 1),
      data: MODEL_BASE + path
    }));

    setStatus("Loading UNET_A and its external weights. This is a multi-gigabyte component; close other memory-heavy tabs and do not start the other UNet tests at the same time.");
    const started = performance.now();
    session = await ort.InferenceSession.create(MODEL_BASE + component.onnx, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "basic",
      externalData
    });

    const expectedInputs = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"];
    const missing = expectedInputs.filter(name => !session.inputNames.includes(name));
    if (missing.length) {
      throw new Error("UNET_A is missing expected inputs: " + missing.join(", ") +
        ". Actual inputs: " + session.inputNames.join(", "));
    }

    const inputMetadata = session.inputMetadata || [];
    const metadataByName = new Map(session.inputNames.map((name, index) => [name, inputMetadata[index]]));
    const expectedTypes = {
      sample: "float16",
      timestep: "float32",
      encoder_hidden_states: "float16",
      text_embeds: "float16",
      time_ids: "float32"
    };
    for (const [name, expectedType] of Object.entries(expectedTypes)) {
      const actualType = metadataByName.get(name)?.type;
      if (actualType && actualType !== expectedType) {
        throw new Error("UNET_A input " + name + " is " + actualType + "; metadata report expected " + expectedType + ".");
      }
    }

    const inputs = {
      sample: new ort.Tensor("float16", new Float16Array(4 * 128 * 128), [1, 4, 128, 128]),
      timestep: new ort.Tensor("float32", new Float32Array([0]), [1]),
      encoder_hidden_states: new ort.Tensor("float16", new Float16Array(77 * 2048), [1, 77, 2048]),
      text_embeds: new ort.Tensor("float16", new Float16Array(1280), [1, 1280]),
      time_ids: new ort.Tensor("float32", new Float32Array(6), [1, 6])
    };

    setStatus("Running a zero-input UNET_A inference on WebGPU. All output tensors will be checked for finite values…");
    const inferenceStarted = performance.now();
    const outputs = await session.run(inputs);
    const outputNames = session.outputNames;
    if (!outputNames.length) throw new Error("UNET_A graph exposes no outputs.");

    let totalValues = 0;
    for (const outputName of outputNames) {
      const output = outputs[outputName];
      if (!output || !output.data || !output.dims) {
        throw new Error("UNET_A returned no readable tensor for output " + outputName + ".");
      }

      for (let i = 0; i < output.data.length; i++) {
        if (!Number.isFinite(Number(output.data[i]))) {
          throw new Error("UNET_A output " + outputName + " contains a non-finite value at element " + i + ".");
        }
      }

      totalValues += output.data.length;
      addRow("Output " + outputName, output.dims.join(" × ") + " · " + output.type + " · all finite");
    }

    addRow("Session inputs", session.inputNames.join(", "));
    addRow("Session outputs", outputNames.join(", "));
    addRow("Output values checked", totalValues.toLocaleString());
    addRow("Inference time", ((performance.now() - inferenceStarted) / 1000).toFixed(1) + " s");
    addRow("Load + inference", ((performance.now() - started) / 1000).toFixed(1) + " s");

    setStatus("UNET_A loaded and completed one zero-input WebGPU inference. This validates only the first partition; it does not yet validate B/C chaining or generate an image.", "success");
  } catch (error) {
    console.error("[Illustrious] UNET_A smoke test failed:", error);
    setStatus("UNET_A WebGPU smoke test failed: " + (error?.message || String(error)), "error");
  } finally {
    if (session) {
      try {
        await session.release();
      } catch (error) {
        console.warn("[Illustrious] Could not release UNET_A session:", error);
      }
    }
    button.disabled = false;
  }
});
