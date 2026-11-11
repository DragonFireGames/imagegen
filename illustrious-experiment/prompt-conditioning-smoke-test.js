import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";
import { AutoTokenizer, env } from "../vendor/transformers/transformers.js";

const MODEL_ID = "aaaaaeeeee/blip-sdxl-anime-wai-illustrious17";
const MODEL_BASE = "https://huggingface.co/" + MODEL_ID + "/resolve/main/";
const MANIFEST_URL = MODEL_BASE + "manifest.json";

ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

// The pack stores its two CLIP tokenizers in different subdirectories.
// Transformers.js supports a configurable remote path template, so load
// each tokenizer from its own subfolder without copying files into the app.
env.allowRemoteModels = true;
env.useBrowserCache = true;

const button = document.querySelector("#testPromptConditioning");
const status = document.querySelector("#promptConditioningStatus");
const results = document.querySelector("#promptConditioningSummary");
const promptInput = document.querySelector("#conditioningPrompt");

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

function finiteFp16(tensor, label, shape) {
  if (!tensor?.data || tensor.type !== "float16" || !tensor.dims) {
    throw new Error(label + " is not a readable FP16 tensor.");
  }
  if (tensor.dims.length !== shape.length ||
      tensor.dims.some((dimension, index) => dimension !== shape[index])) {
    throw new Error(label + " has shape [" + tensor.dims.join(", ") +
      "], expected [" + shape.join(", ") + "].");
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
  return values;
}

async function loadTokenizer(subfolder) {
  env.remotePathTemplate = "{model}/resolve/{revision}/" + subfolder + "/";
  return AutoTokenizer.from_pretrained(MODEL_ID, { revision: "main" });
}

async function loadOnnxComponent(component, label) {
  if (!component?.onnx || !Array.isArray(component.shards) || !component.shards.length) {
    throw new Error("The manifest is missing " + label + " or its external-data shards.");
  }
  const externalData = component.shards.map(path => ({
    path: path.slice(path.lastIndexOf("/") + 1),
    data: MODEL_BASE + path
  }));
  return ort.InferenceSession.create(MODEL_BASE + component.onnx, {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "basic",
    externalData
  });
}

async function waitFrame() {
  return new Promise(resolve => requestAnimationFrame(() =>
    requestAnimationFrame(resolve)
  ));
}

button.addEventListener("click", async () => {
  button.disabled = true;
  results.replaceChildren();
  results.hidden = false;

  let tokenizerL;
  let tokenizerG;
  let sessionL;
  let sessionG;

  try {
    const prompt = promptInput.value.trim();
    if (!prompt) throw new Error("Enter a prompt before testing text conditioning.");
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("The browser did not return a WebGPU adapter.");

    setStatus("Fetching the model manifest…");
    const response = await fetch(MANIFEST_URL, { credentials: "omit" });
    if (!response.ok) throw new Error("Model manifest request failed: HTTP " + response.status);
    const manifest = await response.json();

    addRow("Model pack", manifest.packageId || MODEL_ID);
    addRow("Prompt", prompt);
    addRow("CLIP-L component", ((manifest.components?.text_encoder?.bytes || 0) / 1024 / 1024).toFixed(1) + " MiB");
    addRow("CLIP-bigG component", ((manifest.components?.text_encoder_2?.bytes || 0) / 1024 / 1024).toFixed(1) + " MiB");
    addRow("Execution provider", "WebGPU · one text encoder session at a time");

    setStatus("Loading tokenizer 1 from tokenizer/…");
    const started = performance.now();
    tokenizerL = await loadTokenizer("tokenizer");
    const encodedL = await tokenizerL(prompt, {
      padding: "max_length",
      truncation: true,
      max_length: 77
    });

    setStatus("Loading tokenizer 2 from tokenizer_2/…");
    tokenizerG = await loadTokenizer("tokenizer_2");
    const encodedG = await tokenizerG(prompt, {
      padding: "max_length",
      truncation: true,
      max_length: 77
    });

    const idsL = encodedL.input_ids;
    const idsG = encodedG.input_ids;
    if (!idsL?.data || idsL.type !== "int64" ||
        idsL.dims.length !== 2 || idsL.dims[0] !== 1 || idsL.dims[1] !== 77) {
      throw new Error("CLIP-L tokenizer did not produce int64 input_ids [1, 77].");
    }
    if (!idsG?.data || idsG.type !== "int64" ||
        idsG.dims.length !== 2 || idsG.dims[0] !== 1 || idsG.dims[1] !== 77) {
      throw new Error("CLIP-bigG tokenizer did not produce int64 input_ids [1, 77].");
    }

    const attentionL = encodedL.attention_mask?.data;
    const attentionG = encodedG.attention_mask?.data;
    const countL = attentionL ? Array.from(attentionL).reduce((sum, value) => sum + (Number(value) > 0 ? 1 : 0), 0) : "unavailable";
    const countG = attentionG ? Array.from(attentionG).reduce((sum, value) => sum + (Number(value) > 0 ? 1 : 0), 0) : "unavailable";
    addRow("CLIP-L tokenizer", "int64 [1, 77] · " + countL + " attended positions");
    addRow("CLIP-bigG tokenizer", "int64 [1, 77] · " + countG + " attended positions");
    addRow("CLIP-L token IDs", Array.from(idsL.data).slice(0, 16).map(String).join(", "));
    addRow("CLIP-bigG token IDs", Array.from(idsG.data).slice(0, 16).map(String).join(", "));

    // Stage 1: CLIP-L. Keep its small FP16 hidden-state output on CPU.
    setStatus("Loading CLIP-L and encoding the actual prompt…");
    const loadL = performance.now();
    sessionL = await loadOnnxComponent(manifest.components?.text_encoder, "CLIP-L");
    if (!sessionL.inputNames.includes("input_ids")) {
      throw new Error("CLIP-L graph is missing input_ids.");
    }
    const outputL = await sessionL.run({
      input_ids: new ort.Tensor("int64", idsL.data, [1, 77])
    });
    const hiddenL = finiteFp16(outputL.penultimate_hidden_state,
      "CLIP-L penultimate_hidden_state", [1, 77, 768]);
    addRow("CLIP-L inference", ((performance.now() - loadL) / 1000).toFixed(1) + " s · FP16 [1, 77, 768]");
    outputL.penultimate_hidden_state = null;
    await sessionL.release();
    sessionL = null;
    await waitFrame();

    // Stage 2: CLIP-bigG. Keep penultimate states and pooled text embedding.
    setStatus("Loading CLIP-bigG and encoding the same prompt…");
    const loadG = performance.now();
    sessionG = await loadOnnxComponent(manifest.components?.text_encoder_2, "CLIP-bigG");
    if (!sessionG.inputNames.includes("input_ids")) {
      throw new Error("CLIP-bigG graph is missing input_ids.");
    }
    const outputG = await sessionG.run({
      input_ids: new ort.Tensor("int64", idsG.data, [1, 77])
    });
    const hiddenG = finiteFp16(outputG.penultimate_hidden_state,
      "CLIP-bigG penultimate_hidden_state", [1, 77, 1280]);
    const pooledG = finiteFp16(outputG.text_embeds, "CLIP-bigG text_embeds", [1, 1280]);
    addRow("CLIP-bigG inference", ((performance.now() - loadG) / 1000).toFixed(1) + " s · FP16 [1, 77, 1280] + [1, 1280]");
    await sessionG.release();
    sessionG = null;
    await waitFrame();

    // SDXL combines CLIP-L and CLIP-bigG penultimate hidden states per token.
    const combined = new Float16Array(77 * 2048);
    for (let token = 0; token < 77; token++) {
      combined.set(hiddenL.subarray(token * 768, token * 768 + 768), token * 2048);
      combined.set(hiddenG.subarray(token * 1280, token * 1280 + 1280), token * 2048 + 768);
    }

    for (let i = 0; i < combined.length; i++) {
      if (!Number.isFinite(combined[i])) throw new Error("Combined SDXL conditioning contains a non-finite value.");
    }
    addRow("Combined SDXL conditioning", "FP16 [1, 77, 2048] · all " + combined.length.toLocaleString() + " values finite");
    addRow("Pooled text embedding", "FP16 [1, 1280] · all " + pooledG.length.toLocaleString() + " values finite");
    addRow("Total tokenization + load + inference", ((performance.now() - started) / 1000).toFixed(1) + " s");

    setStatus("Real prompt tokenization and both CLIP encoders passed. The test assembled the SDXL conditioning tensors [1, 77, 2048] and [1, 1280]. No UNet or VAE was loaded and no image was generated.", "success");
  } catch (error) {
    console.error("[Illustrious] Prompt conditioning smoke test failed:", error);
    setStatus("Prompt conditioning test failed: " + (error?.message || String(error)), "error");
  } finally {
    if (sessionL) {
      try { await sessionL.release(); } catch (error) {
        console.warn("[Illustrious] Could not release CLIP-L session:", error);
      }
    }
    if (sessionG) {
      try { await sessionG.release(); } catch (error) {
        console.warn("[Illustrious] Could not release CLIP-bigG session:", error);
      }
    }
    button.disabled = false;
  }
});
