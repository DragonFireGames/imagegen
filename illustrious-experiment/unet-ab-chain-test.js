import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";

const MODEL_BASE = "https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/resolve/main/";
const MANIFEST_URL = MODEL_BASE + "manifest.json";

ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

const button = document.querySelector("#testUnetAB");
const status = document.querySelector("#unetABStatus");
const results = document.querySelector("#unetABSummary");

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

function fp16Values(tensor, label) {
  if (!tensor?.data || tensor.type !== "float16") {
    throw new Error(label + " is missing or is not a readable FP16 tensor.");
  }
  if (!(tensor.data instanceof Float16Array) && !(tensor.data instanceof Uint16Array)) {
    throw new Error(label + " data is " + tensor.data.constructor.name +
      "; expected a 16-bit float backing array.");
  }

  // ONNX Runtime Web exposes float16 CPU data as Float16Array when available,
  // or Uint16Array containing raw IEEE-754 half bits otherwise. Copy the bytes
  // rather than converting numeric values, which would corrupt raw half bits.
  const bytes = tensor.data.buffer.slice(
    tensor.data.byteOffset,
    tensor.data.byteOffset + tensor.data.byteLength
  );
  const values = new Float16Array(bytes);

  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new Error(label + " contains a non-finite FP16 value at element " + i + ".");
    }
  }
  return values;
}

function copyFp16(tensor, name, expectedShape) {
  if (!tensor?.dims || tensor.dims.length !== expectedShape.length ||
      tensor.dims.some((dimension, index) => dimension !== expectedShape[index])) {
    const actual = tensor?.dims ? tensor.dims.join(", ") : "missing";
    throw new Error("A." + name + " has shape [" + actual +
      "], expected [" + expectedShape.join(", ") + "].");
  }
  const values = fp16Values(tensor, "A." + name);
  const expectedSize = expectedShape.reduce((total, dimension) => total * dimension, 1);
  if (values.length !== expectedSize) {
    throw new Error("A." + name + " has " + values.length + " values; expected " + expectedSize + ".");
  }
  return values;
}

function validateOutput(output, name, shape) {
  if (!output?.dims || output.dims.length !== shape.length ||
      output.dims.some((dimension, index) => dimension !== shape[index])) {
    const actual = output?.dims ? output.dims.join(", ") : "missing";
    throw new Error("B." + name + " has shape [" + actual +
      "], expected [" + shape.join(", ") + "].");
  }
  fp16Values(output, "B." + name);
  const expectedSize = shape.reduce((total, dimension) => total * dimension, 1);
  if (output.data.length !== expectedSize) {
    throw new Error("B." + name + " has " + output.data.length + " values; expected " + expectedSize + ".");
  }
}

button.addEventListener("click", async () => {
  button.disabled = true;
  results.replaceChildren();
  results.hidden = false;

  let aSession;
  let bSession;
  let staged = null;

  try {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("The browser did not return a WebGPU adapter.");

    const manifest = await loadManifest();
    const componentA = manifest?.components?.unet_a;
    const componentB = manifest?.components?.unet_b;
    if (!componentA?.onnx || !componentA.shards?.length) {
      throw new Error("The model manifest is missing UNET_A or its external-data shards.");
    }
    if (!componentB?.onnx || !componentB.shards?.length) {
      throw new Error("The model manifest is missing UNET_B or its external-data shards.");
    }

    addRow("Model pack", manifest.packageId || "WAI Illustrious v17");
    addRow("Execution provider", "WebGPU");
    addRow("Resolution", "1024 × 1024 · latent 128 × 128");
    addRow("UNET_A weights", formatBytes(componentA.bytes));
    addRow("UNET_B weights", formatBytes(componentB.bytes));
    addRow("Memory strategy", "A runs first; selected FP16 outputs are copied to CPU memory; A session is released before B loads.");

    const started = performance.now();
    let stageAStarted = performance.now();

    const externalA = componentA.shards.map(path => ({
      path: path.slice(path.lastIndexOf("/") + 1),
      data: MODEL_BASE + path
    }));

    setStatus("Stage 1/2: loading UNET_A. This is a multi-gigabyte allocation; no other UNet session is created.");
    aSession = await ort.InferenceSession.create(MODEL_BASE + componentA.onnx, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "basic",
      externalData: externalA
    });

    const inputsA = {
      sample: new ort.Tensor("float16", new Float16Array(4 * 128 * 128), [1, 4, 128, 128]),
      timestep: new ort.Tensor("float32", new Float32Array([0]), [1]),
      encoder_hidden_states: new ort.Tensor("float16", new Float16Array(77 * 2048), [1, 77, 2048]),
      text_embeds: new ort.Tensor("float16", new Float16Array(1280), [1, 1280]),
      time_ids: new ort.Tensor("float32", new Float32Array(6), [1, 6])
    };

    setStatus("Stage 1/2: running UNET_A and copying only B's required intermediate tensors…");
    const outputsA = await aSession.run(inputsA);
    const requiredA = {
      hidden: [1, 1280, 32, 32],
      emb: [1, 1280],
      skip_6: [1, 640, 32, 32],
      skip_7: [1, 1280, 32, 32],
      skip_8: [1, 1280, 32, 32]
    };

    staged = {};
    let retainedValues = 0;
    for (const [name, shape] of Object.entries(requiredA)) {
      staged[name] = copyFp16(outputsA[name], name, shape);
      retainedValues += staged[name].length;
    }

    const timeA = (performance.now() - stageAStarted) / 1000;
    addRow("UNET_A", "Inference passed; copied hidden, emb, skip_6, skip_7, skip_8.");
    addRow("Intermediate data retained", retainedValues.toLocaleString() + " FP16 values · " +
      formatBytes(retainedValues * 2));
    addRow("UNET_A load + inference", timeA.toFixed(1) + " s");

    // Release A before loading B. The only A-derived data kept is the small set
    // of CPU-side FP16 tensors that B consumes.
    setStatus("Releasing UNET_A before creating the UNET_B session…");
    await aSession.release();
    aSession = null;

    // Do not retain A's other outputs or its run result while B allocates.
    for (const key of Object.keys(outputsA)) delete outputsA[key];

    await new Promise(resolve => requestAnimationFrame(() => resolve()));
    await new Promise(resolve => requestAnimationFrame(() => resolve()));

    const externalB = componentB.shards.map(path => ({
      path: path.slice(path.lastIndexOf("/") + 1),
      data: MODEL_BASE + path
    }));

    const stageBStarted = performance.now();
    setStatus("Stage 2/2: loading UNET_B after UNET_A's session has been released. This is another multi-gigabyte allocation.");
    bSession = await ort.InferenceSession.create(MODEL_BASE + componentB.onnx, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "basic",
      externalData: externalB
    });

    const requiredInputsB = ["hidden", "emb", "encoder_hidden_states", "skip_6", "skip_7", "skip_8"];
    const missingInputs = requiredInputsB.filter(name => !bSession.inputNames.includes(name));
    if (missingInputs.length) {
      throw new Error("UNET_B is missing expected inputs: " + missingInputs.join(", ") +
        ". Actual inputs: " + bSession.inputNames.join(", "));
    }

    const metadataB = bSession.inputMetadata || [];
    const byNameB = new Map(bSession.inputNames.map((name, index) => [name, metadataB[index]]));
    for (const name of requiredInputsB) {
      const type = byNameB.get(name)?.type;
      if (type && type !== "float16") {
        throw new Error("UNET_B input " + name + " is " + type + "; expected float16.");
      }
    }

    setStatus("Stage 2/2: feeding the actual UNET_A intermediates into UNET_B…");
    const inputsB = {
      hidden: new ort.Tensor("float16", staged.hidden, [1, 1280, 32, 32]),
      emb: new ort.Tensor("float16", staged.emb, [1, 1280]),
      encoder_hidden_states: new ort.Tensor("float16", new Float16Array(77 * 2048), [1, 77, 2048]),
      skip_6: new ort.Tensor("float16", staged.skip_6, [1, 640, 32, 32]),
      skip_7: new ort.Tensor("float16", staged.skip_7, [1, 1280, 32, 32]),
      skip_8: new ort.Tensor("float16", staged.skip_8, [1, 1280, 32, 32])
    };

    const outputsB = await bSession.run(inputsB);
    const outputName = "hidden_out";
    if (!bSession.outputNames.includes(outputName)) {
      throw new Error("UNET_B does not expose hidden_out. Actual outputs: " + bSession.outputNames.join(", "));
    }
    const outputB = outputsB[outputName];
    validateOutput(outputB, outputName, [1, 1280, 64, 64]);

    addRow("UNET_B input", "Actual UNET_A intermediates + zero FP16 encoder_hidden_states.");
    addRow("UNET_B session inputs", bSession.inputNames.join(", "));
    addRow("UNET_B session outputs", bSession.outputNames.join(", "));
    addRow("UNET_B hidden_out", outputB.dims.join(" × ") + " · " + outputB.type + " · all finite");
    addRow("UNET_B load + inference", ((performance.now() - stageBStarted) / 1000).toFixed(1) + " s");
    addRow("Total A → B time", ((performance.now() - started) / 1000).toFixed(1) + " s");

    setStatus("UNET_A → UNET_B chaining passed. B consumed the actual A intermediate tensors and returned finite FP16 hidden_out [1, 1280, 64, 64]. UNET_C and the final denoiser output remain untested.", "success");
  } catch (error) {
    console.error("[Illustrious] UNET_A → UNET_B chain test failed:", error);
    setStatus("UNET_A → UNET_B chain test failed: " + (error?.message || String(error)), "error");
  } finally {
    staged = null;
    if (aSession) {
      try {
        await aSession.release();
      } catch (error) {
        console.warn("[Illustrious] Could not release UNET_A session:", error);
      }
    }
    if (bSession) {
      try {
        await bSession.release();
      } catch (error) {
        console.warn("[Illustrious] Could not release UNET_B session:", error);
      }
    }
    button.disabled = false;
  }
});
