import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";
import { acquireHeavyModelTest, releaseHeavyModelTest } from "./heavy-model-test-lock.js";

const MODEL_BASE = "https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/resolve/main/";
const MANIFEST_URL = MODEL_BASE + "manifest.json";

ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

const button = document.querySelector("#testUnetABC");
const status = document.querySelector("#unetABCStatus");
const results = document.querySelector("#unetABCSummary");

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

function fp16Copy(tensor, label, expectedShape) {
  if (!tensor?.data || tensor.type !== "float16" || !tensor.dims) {
    throw new Error(label + " is missing or is not a readable FP16 tensor.");
  }
  if (tensor.dims.length !== expectedShape.length ||
      tensor.dims.some((dimension, index) => dimension !== expectedShape[index])) {
    throw new Error(label + " has shape [" + tensor.dims.join(", ") +
      "], expected [" + expectedShape.join(", ") + "].");
  }
  if (!(tensor.data instanceof Float16Array) && !(tensor.data instanceof Uint16Array)) {
    throw new Error(label + " backing data is " + tensor.data.constructor.name +
      "; expected Float16Array or Uint16Array.");
  }

  // Preserve raw IEEE-754 half bits regardless of ORT's Float16Array/Uint16Array
  // CPU representation. Numeric conversion of a Uint16Array would corrupt it.
  const buffer = tensor.data.buffer.slice(
    tensor.data.byteOffset,
    tensor.data.byteOffset + tensor.data.byteLength
  );
  const values = new Float16Array(buffer);
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new Error(label + " contains a non-finite value at element " + i + ".");
    }
  }
  const expectedSize = expectedShape.reduce((total, dimension) => total * dimension, 1);
  if (values.length !== expectedSize) {
    throw new Error(label + " contains " + values.length + " values; expected " + expectedSize + ".");
  }
  return values;
}

function checkInputs(session, expected, component) {
  const missing = expected.filter(name => !session.inputNames.includes(name));
  if (missing.length) {
    throw new Error(component + " is missing inputs " + missing.join(", ") +
      ". Actual inputs: " + session.inputNames.join(", "));
  }
  const metadata = session.inputMetadata || [];
  const byName = new Map(session.inputNames.map((name, index) => [name, metadata[index]]));
  for (const [name, type] of Object.entries({
    sample: "float16",
    timestep: "float32",
    encoder_hidden_states: "float16",
    text_embeds: "float16",
    time_ids: "float32",
    hidden: "float16",
    emb: "float16",
    skip_0: "float16",
    skip_1: "float16",
    skip_2: "float16",
    skip_3: "float16",
    skip_4: "float16",
    skip_5: "float16",
    skip_6: "float16",
    skip_7: "float16",
    skip_8: "float16"
  })) {
    if (!expected.includes(name)) continue;
    const actual = byName.get(name)?.type;
    if (actual && actual !== type) {
      throw new Error(component + " input " + name + " is " + actual + "; expected " + type + ".");
    }
  }
}

function externalData(component) {
  return component.shards.map(path => ({
    path: path.slice(path.lastIndexOf("/") + 1),
    data: MODEL_BASE + path
  }));
}

function yieldFrames() {
  return new Promise(resolve => requestAnimationFrame(() =>
    requestAnimationFrame(resolve)
  ));
}

async function releaseSession(session, label) {
  if (!session) return;
  await session.release();
  await yieldFrames();
  setStatus(label + " released; preparing the next partition…");
}

button.addEventListener("click", async () => {
  button.disabled = true;
  results.replaceChildren();
  results.hidden = false;

  let aSession = null;
  let bSession = null;
  let cSession = null;
  let inputsA = null;
  let outputsA = null;
  let inputsB = null;
  let outputsB = null;
  let inputsC = null;
  let outputsC = null;
  let aToB = null;
  let aToC = null;
  let hiddenFromB = null;

  let lockAcquired = false;
  try {
    acquireHeavyModelTest("UNET_A → UNET_B → UNET_C");
    lockAcquired = true;
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("The browser did not return a WebGPU adapter.");

    const manifest = await loadManifest();
    const parts = {
      a: manifest?.components?.unet_a,
      b: manifest?.components?.unet_b,
      c: manifest?.components?.unet_c
    };
    for (const [name, component] of Object.entries(parts)) {
      if (!component?.onnx || !Array.isArray(component.shards) || !component.shards.length) {
        throw new Error("The model manifest is missing UNET_" + name.toUpperCase() + " or its shards.");
      }
    }

    addRow("Model pack", manifest.packageId || "WAI Illustrious v17");
    addRow("Execution provider", "WebGPU");
    addRow("Resolution", "1024 × 1024 · latent 128 × 128");
    addRow("UNET_A weights", formatBytes(parts.a.bytes));
    addRow("UNET_B weights", formatBytes(parts.b.bytes));
    addRow("UNET_C weights", formatBytes(parts.c.bytes));
    addRow("Session strategy", "One partition resident at a time; only required FP16 intermediates are copied to CPU memory.");

    const started = performance.now();
    let stageStarted = performance.now();

    // Stage A: down path + middle.
    setStatus("Stage 1/3: loading UNET_A. No other UNet session is resident.");
    aSession = await ort.InferenceSession.create(MODEL_BASE + parts.a.onnx, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "basic",
      externalData: externalData(parts.a)
    });
    checkInputs(aSession, ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"], "UNET_A");

    inputsA = {
      sample: new ort.Tensor("float16", new Float16Array(4 * 128 * 128), [1, 4, 128, 128]),
      timestep: new ort.Tensor("float32", new Float32Array([0]), [1]),
      encoder_hidden_states: new ort.Tensor("float16", new Float16Array(77 * 2048), [1, 77, 2048]),
      text_embeds: new ort.Tensor("float16", new Float16Array(1280), [1, 1280]),
      time_ids: new ort.Tensor("float32", new Float32Array(6), [1, 6])
    };

    setStatus("Stage 1/3: running UNET_A and copying only tensors required by B and C…");
    outputsA = await aSession.run(inputsA);

    const aEmb = fp16Copy(outputsA.emb, "A.emb", [1, 1280]);
    aToB = {
      hidden: fp16Copy(outputsA.hidden, "A.hidden", [1, 1280, 32, 32]),
      emb: aEmb,
      skip_6: fp16Copy(outputsA.skip_6, "A.skip_6", [1, 640, 32, 32]),
      skip_7: fp16Copy(outputsA.skip_7, "A.skip_7", [1, 1280, 32, 32]),
      skip_8: fp16Copy(outputsA.skip_8, "A.skip_8", [1, 1280, 32, 32])
    };
    aToC = {
      emb: aEmb,
      skip_0: fp16Copy(outputsA.skip_0, "A.skip_0", [1, 320, 128, 128]),
      skip_1: fp16Copy(outputsA.skip_1, "A.skip_1", [1, 320, 128, 128]),
      skip_2: fp16Copy(outputsA.skip_2, "A.skip_2", [1, 320, 128, 128]),
      skip_3: fp16Copy(outputsA.skip_3, "A.skip_3", [1, 320, 64, 64]),
      skip_4: fp16Copy(outputsA.skip_4, "A.skip_4", [1, 640, 64, 64]),
      skip_5: fp16Copy(outputsA.skip_5, "A.skip_5", [1, 640, 64, 64])
    };

    const aValues = Object.values(aToB).reduce((sum, tensor) => sum + tensor.length, 0) +
      Object.values(aToC).reduce((sum, tensor) => sum + tensor.length, 0) - aEmb.length;
    addRow("UNET_A", "Passed; copied B/C intermediates (hidden, emb, skip_0–8).");
    addRow("A intermediates retained", aValues.toLocaleString() + " FP16 values · " + formatBytes(aValues * 2));
    addRow("UNET_A load + inference", ((performance.now() - stageStarted) / 1000).toFixed(1) + " s");

    inputsA = null;
    outputsA = null;
    await releaseSession(aSession, "UNET_A");
    aSession = null;

    // Stage B: first up path. Keep A's CPU-side skip_0–5 for C.
    stageStarted = performance.now();
    setStatus("Stage 2/3: loading UNET_B after UNET_A has been released.");
    bSession = await ort.InferenceSession.create(MODEL_BASE + parts.b.onnx, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "basic",
      externalData: externalData(parts.b)
    });
    checkInputs(bSession, ["hidden", "emb", "encoder_hidden_states", "skip_6", "skip_7", "skip_8"], "UNET_B");

    inputsB = {
      hidden: new ort.Tensor("float16", aToB.hidden, [1, 1280, 32, 32]),
      emb: new ort.Tensor("float16", aToB.emb, [1, 1280]),
      encoder_hidden_states: new ort.Tensor("float16", new Float16Array(77 * 2048), [1, 77, 2048]),
      skip_6: new ort.Tensor("float16", aToB.skip_6, [1, 640, 32, 32]),
      skip_7: new ort.Tensor("float16", aToB.skip_7, [1, 1280, 32, 32]),
      skip_8: new ort.Tensor("float16", aToB.skip_8, [1, 1280, 32, 32])
    };

    setStatus("Stage 2/3: running UNET_B on the actual UNET_A intermediates…");
    outputsB = await bSession.run(inputsB);
    hiddenFromB = fp16Copy(outputsB.hidden_out, "B.hidden_out", [1, 1280, 64, 64]);
    addRow("UNET_B", "Passed; consumed A's actual hidden, emb, and skip_6–8 tensors.");
    addRow("UNET_B hidden_out", "1 × 1280 × 64 × 64 · float16 · all finite");
    addRow("UNET_B load + inference", ((performance.now() - stageStarted) / 1000).toFixed(1) + " s");

    inputsB = null;
    outputsB = null;
    aToB = null;
    await releaseSession(bSession, "UNET_B");
    bSession = null;

    // Stage C: remaining up path. Its inputs combine B.hidden_out with A's
    // retained skip_0..5 and emb. Only C's own weights are resident now.
    stageStarted = performance.now();
    setStatus("Stage 3/3: loading UNET_C after UNET_B has been released.");
    cSession = await ort.InferenceSession.create(MODEL_BASE + parts.c.onnx, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "basic",
      externalData: externalData(parts.c)
    });
    checkInputs(cSession, [
      "hidden", "emb", "encoder_hidden_states",
      "skip_0", "skip_1", "skip_2", "skip_3", "skip_4", "skip_5"
    ], "UNET_C");

    inputsC = {
      hidden: new ort.Tensor("float16", hiddenFromB, [1, 1280, 64, 64]),
      emb: new ort.Tensor("float16", aToC.emb, [1, 1280]),
      encoder_hidden_states: new ort.Tensor("float16", new Float16Array(77 * 2048), [1, 77, 2048]),
      skip_0: new ort.Tensor("float16", aToC.skip_0, [1, 320, 128, 128]),
      skip_1: new ort.Tensor("float16", aToC.skip_1, [1, 320, 128, 128]),
      skip_2: new ort.Tensor("float16", aToC.skip_2, [1, 320, 128, 128]),
      skip_3: new ort.Tensor("float16", aToC.skip_3, [1, 320, 64, 64]),
      skip_4: new ort.Tensor("float16", aToC.skip_4, [1, 640, 64, 64]),
      skip_5: new ort.Tensor("float16", aToC.skip_5, [1, 640, 64, 64])
    };

    setStatus("Stage 3/3: running UNET_C on B.hidden_out and A's remaining skip tensors…");
    outputsC = await cSession.run(inputsC);
    const outputName = "out_sample";
    const outputNames = cSession.outputNames;
    if (!outputNames.includes(outputName)) {
      throw new Error("UNET_C does not expose out_sample. Actual outputs: " + outputNames.join(", "));
    }
    const denoised = fp16Copy(outputsC[outputName], "C.out_sample", [1, 4, 128, 128]);

    addRow("UNET_C", "Passed; consumed B.hidden_out and A's emb + skip_0–5.");
    addRow("UNET_C out_sample", "1 × 4 × 128 × 128 · float16 · all finite");
    addRow("Output values checked", denoised.length.toLocaleString());
    addRow("UNET_C load + inference", ((performance.now() - stageStarted) / 1000).toFixed(1) + " s");
    addRow("Total A → B → C", ((performance.now() - started) / 1000).toFixed(1) + " s");

    setStatus("UNET_A → UNET_B → UNET_C chaining passed. The three partitions produced a finite FP16 denoiser output [1, 4, 128, 128]. This is a zero-conditioning forward pass, not a generated image.", "success");
  } catch (error) {
    console.error("[Illustrious] UNET_A → UNET_B → UNET_C chain test failed:", error);
    setStatus("UNET_A → UNET_B → UNET_C chain test failed: " + (error?.message || String(error)), "error");
  } finally {
    inputsA = outputsA = inputsB = outputsB = inputsC = outputsC = null;
    aToB = aToC = hiddenFromB = null;
    for (const [session, label] of [[aSession, "UNET_A"], [bSession, "UNET_B"], [cSession, "UNET_C"]]) {
      if (session) {
        try {
          await session.release();
        } catch (error) {
          console.warn("[Illustrious] Could not release " + label + " session:", error);
        }
      }
    }
    if (lockAcquired) releaseHeavyModelTest("UNET_A → UNET_B → UNET_C");
    button.disabled = false;
  }
});
