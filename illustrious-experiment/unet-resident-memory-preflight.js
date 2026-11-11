import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";
import { acquireHeavyModelTest, releaseHeavyModelTest } from "./heavy-model-test-lock.js";

const MODEL_BASE = "https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/resolve/main/";
const MANIFEST_URL = MODEL_BASE + "manifest.json";

ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

const button = document.querySelector("#testUnetResident");
const status = document.querySelector("#unetResidentStatus");
const results = document.querySelector("#unetResidentSummary");

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
  if (bytes >= 1024 ** 2) return (bytes / 1024 ** 2).toFixed(1) + " MiB";
  return (bytes / 1024).toFixed(1) + " KiB";
}

async function yieldFrames() {
  await new Promise(resolve => requestAnimationFrame(() =>
    requestAnimationFrame(resolve)
  ));
}

async function releaseSession(session, label) {
  if (!session) return;
  await session.release();
  await yieldFrames();
  setStatus(label + " released.");
}

function validateInterface(session, expectedInputs, expectedOutputs, label) {
  const missingInputs = expectedInputs.filter(name => !session.inputNames.includes(name));
  const missingOutputs = expectedOutputs.filter(name => !session.outputNames.includes(name));
  if (missingInputs.length || missingOutputs.length) {
    throw new Error(label + " interface mismatch. Missing inputs: " +
      (missingInputs.join(", ") || "none") + "; missing outputs: " +
      (missingOutputs.join(", ") || "none") + ".");
  }
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

  const sessions = [];
  let lockAcquired = false;
  let loadedBytes = 0;

  try {
    acquireHeavyModelTest("UNet resident-memory preflight");
    lockAcquired = true;

    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("The browser did not return a WebGPU adapter.");

    const manifest = await loadManifest();
    const parts = [
      {
        key: "unet_a",
        label: "UNET_A",
        inputs: ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
        outputs: ["hidden", "emb", "skip_0", "skip_1", "skip_2", "skip_3", "skip_4", "skip_5", "skip_6", "skip_7", "skip_8"]
      },
      {
        key: "unet_b",
        label: "UNET_B",
        inputs: ["hidden", "emb", "encoder_hidden_states", "skip_6", "skip_7", "skip_8"],
        outputs: ["hidden_out"]
      },
      {
        key: "unet_c",
        label: "UNET_C",
        inputs: ["hidden", "emb", "encoder_hidden_states", "skip_0", "skip_1", "skip_2", "skip_3", "skip_4", "skip_5"],
        outputs: ["out_sample"]
      }
    ];

    for (const part of parts) {
      part.component = manifest?.components?.[part.key];
      if (!part.component?.onnx || !Array.isArray(part.component.shards) ||
          !part.component.shards.length) {
        throw new Error("The manifest is missing " + part.label + " or its external-data shards.");
      }
    }

    const totalBytes = parts.reduce((sum, part) => sum + Number(part.component.bytes || 0), 0);
    addRow("Model pack", manifest.packageId || "WAI Illustrious v17");
    addRow("Execution provider", "WebGPU");
    addRow("Partition payload total", formatBytes(totalBytes));
    addRow("Memory strategy", "Keep A resident while creating B, then keep A+B resident while creating C.");
    addRow("Inference", "Not run · this is a session-residency preflight");
    addRow("Per-buffer limit", formatBytes(adapter.limits?.maxBufferSize || 0));
    addRow("Warning", "The combined model payload is not a measurement of actual GPU memory use. This test may fail due to device/browser memory limits.");

    const started = performance.now();
    for (const part of parts) {
      const stageStarted = performance.now();
      setStatus("Loading " + part.label + " while keeping " +
        (sessions.length ? sessions.map(item => item.label).join(" + ") + " resident" : "no earlier UNet session") +
        ". This may allocate several GiB…");

      const externalData = part.component.shards.map(path => ({
        path: path.slice(path.lastIndexOf("/") + 1),
        data: MODEL_BASE + path
      }));
      const session = await ort.InferenceSession.create(MODEL_BASE + part.component.onnx, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "basic",
        externalData
      });

      validateInterface(session, part.inputs, part.outputs, part.label);
      sessions.push({ label: part.label, session });
      loadedBytes += Number(part.component.bytes || 0);
      addRow(part.label + " session", "Created; " + formatBytes(part.component.bytes) +
        " · " + part.component.shards.length + " external-data shards · required interface present");
      addRow(part.label + " create time", ((performance.now() - stageStarted) / 1000).toFixed(1) + " s");
    }

    addRow("All sessions resident", "UNET_A + UNET_B + UNET_C · " + formatBytes(loadedBytes) + " manifest payload");
    addRow("Total session creation", ((performance.now() - started) / 1000).toFixed(1) + " s");
    setStatus("All three UNet sessions were created and remained resident simultaneously. This validates session allocation only; it does not run inference or prove that a repeated denoising loop fits its activation-memory peak.", "success");
  } catch (error) {
    console.error("[Illustrious] UNet resident-memory preflight failed:", error);
    addRow("Sessions created before failure", sessions.map(item => item.label).join(", ") || "None");
    addRow("Manifest payload represented by created sessions", formatBytes(loadedBytes));
    setStatus("UNet resident-memory preflight failed: " + (error?.message || String(error)) +
      ". Any sessions created by this test will now be released.", "error");
  } finally {
    for (const item of sessions.slice().reverse()) {
      try {
        await releaseSession(item.session, item.label);
      } catch (error) {
        console.warn("[Illustrious] Could not release " + item.label + " session:", error);
      }
    }
    if (lockAcquired) releaseHeavyModelTest("UNet resident-memory preflight");
    button.disabled = false;
  }
});
