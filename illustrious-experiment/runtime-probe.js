import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";

ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
ort.env.wasm.numThreads = 1;

const button = document.querySelector("#probeRuntime");
const status = document.querySelector("#probeStatus");
const results = document.querySelector("#runtimeProbeSummary");

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

function formatLimit(value) {
  if (!Number.isFinite(value) || value <= 0) return "Not reported";
  return value >= 1024 ** 3
    ? (value / 1024 ** 3).toFixed(2) + " GiB"
    : (value / 1024 ** 2).toFixed(0) + " MiB";
}

button.addEventListener("click", async () => {
  button.disabled = true;
  results.replaceChildren();
  results.hidden = false;
  setStatus("Checking the local ONNX Runtime module and browser GPU…");

  try {
    const version = ort.env.versions?.web || "version not exposed";
    addRow("ONNX Runtime Web", version);
    addRow(
      "ORT LoRA adapter API",
      typeof ort.LoraAdapter === "function" ? "Exported by this build" : "Not exported by this build"
    );
    addRow(
      "Float16Array",
      typeof Float16Array === "function" ? "Available" : "Not available"
    );

    if (!navigator.gpu) {
      addRow("WebGPU", "Not exposed by this browser");
      setStatus("The ORT module loaded, but this browser does not expose WebGPU.", "error");
      return;
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      addRow("WebGPU", "No adapter returned");
      setStatus("The browser exposes WebGPU, but no GPU adapter was returned.", "error");
      return;
    }

    const limits = adapter.limits || {};
    addRow("WebGPU", "Adapter available");
    addRow("Maximum buffer size", formatLimit(limits.maxBufferSize));
    addRow("Maximum storage buffer binding", formatLimit(limits.maxStorageBufferBindingSize));
    addRow("Maximum compute workgroups / dimension", String(limits.maxComputeWorkgroupsPerDimension || "Not reported"));

    setStatus(
      "Probe complete. This checks browser capabilities only; it does not load a model, test an ONNX session, or validate LoRA compatibility.",
      "success"
    );
  } catch (error) {
    setStatus("Capability check failed: " + (error?.message || String(error)), "error");
  } finally {
    button.disabled = false;
  }
});
