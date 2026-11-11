const MODEL_BASE = "https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/resolve/main/";
const MANIFEST_URL = MODEL_BASE + "manifest.json";
const HEADER_LIMIT = 64 * 1024 * 1024;

const fileInput = document.querySelector("#loraFile");
const button = document.querySelector("#testLoraPayload");
const status = document.querySelector("#loraPayloadStatus");
const summary = document.querySelector("#loraPayloadSummary");
const output = document.querySelector("#loraPayloadOutput");

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
  summary.append(row);
}

function readBf16(buffer, label) {
  if (buffer.byteLength % 2 !== 0) {
    throw new Error(label + " has an odd byte count; BF16 tensors must use two bytes per element.");
  }
  const input = new DataView(buffer);
  const bits = new Uint32Array(buffer.byteLength / 2);
  for (let i = 0; i < bits.length; i++) {
    bits[i] = input.getUint16(i * 2, true) << 16;
  }
  const values = new Float32Array(bits.buffer);
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new Error(label + " contains a non-finite BF16 value at element " + i + ".");
    }
  }
  return values;
}

async function readSafetensorsHeader(file) {
  if (!file || file.size < 8) throw new Error("Choose a valid .safetensors file first.");
  const prefix = await file.slice(0, 8).arrayBuffer();
  const headerLength = new DataView(prefix).getBigUint64(0, true);
  if (headerLength === 0n || headerLength > BigInt(HEADER_LIMIT)) {
    throw new Error("Safetensors header is invalid or exceeds the 64 MiB safety limit.");
  }
  if (headerLength > BigInt(file.size - 8)) throw new Error("File ends before its declared header.");
  const dataStart = 8 + Number(headerLength);
  const bytes = await file.slice(8, dataStart).arrayBuffer();
  const header = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  return { header, dataStart };
}

function findRepresentativeLinearPair(header, index) {
  if (!index?.entries) throw new Error("CLIP-L weights index has no entries.");
  const indexByModule = new Map();
  for (const [moduleName, entry] of Object.entries(index.entries)) {
    const key = (moduleName.endsWith(".weight")
      ? moduleName.slice(0, -7)
      : moduleName).replaceAll(".", "_");
    indexByModule.set(key, { moduleName, entry });
  }

  const pairs = new Map();
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    let side = null;
    let stem = name;
    if (name.endsWith(".lora_down.weight")) {
      side = "down";
      stem = name.slice(0, -".lora_down.weight".length);
    } else if (name.endsWith(".lora_up.weight")) {
      side = "up";
      stem = name.slice(0, -".lora_up.weight".length);
    }
    if (!side || !stem.startsWith("lora_te1_")) continue;
    if (!pairs.has(stem)) pairs.set(stem, { down: null, up: null, downName: null, upName: null });
    pairs.get(stem)[side] = info;
    pairs.get(stem)[side + "Name"] = name;
  }

  for (const [stem, pair] of pairs) {
    if (!pair.down || !pair.up) continue;
    if (pair.down.dtype !== "BF16" || pair.up.dtype !== "BF16") continue;
    if (pair.down.shape.length !== 2 || pair.up.shape.length !== 2) continue;
    if (pair.down.shape[0] !== pair.up.shape[1]) continue;

    const moduleKey = stem.slice("lora_te1_".length);
    const target = indexByModule.get(moduleKey);
    if (!target || !Array.isArray(target.entry.torchShape) || target.entry.torchShape.length !== 2) continue;

    const [outFeatures, inFeatures] = target.entry.torchShape;
    const [rank, downInput] = pair.down.shape;
    const [upOutput, upRank] = pair.up.shape;
    if (downInput !== inFeatures || upOutput !== outFeatures || upRank !== rank) continue;

    const alphaName = stem + ".alpha";
    const alphaInfo = header[alphaName];
    if (!alphaInfo || alphaInfo.dtype !== "BF16" ||
        !Array.isArray(alphaInfo.shape) || alphaInfo.shape.length !== 0) continue;

    return { stem, pair, alphaName, alphaInfo, target };
  }

  throw new Error("No shape-compatible BF16 CLIP-L linear LoRA pair with a scalar alpha entry was found in the selected file.");
}

async function readTensorPayload(file, dataStart, info, name) {
  if (!Array.isArray(info.data_offsets) || info.data_offsets.length !== 2) {
    throw new Error("Missing data offsets for " + name + ".");
  }
  const start = info.data_offsets[0];
  const end = info.data_offsets[1];
  const expectedElements = info.shape.reduce((product, dimension) => product * dimension, 1);
  const expectedBytes = expectedElements * 2;
  if (end - start !== expectedBytes) {
    throw new Error(name + " byte length mismatch: expected " + expectedBytes + ", found " + (end - start) + ".");
  }
  const buffer = await file.slice(dataStart + start, dataStart + end).arrayBuffer();
  if (buffer.byteLength !== expectedBytes) throw new Error("Truncated local payload for " + name + ".");
  return readBf16(buffer, name);
}

function applyLinearLora(down, up, input, rank, outputSize, scale) {
  const lowRank = new Float32Array(rank);
  for (let r = 0; r < rank; r++) {
    let sum = 0;
    const offset = r * input.length;
    for (let i = 0; i < input.length; i++) sum += down[offset + i] * input[i];
    lowRank[r] = sum;
  }

  const result = new Float32Array(outputSize);
  for (let o = 0; o < outputSize; o++) {
    let sum = 0;
    const offset = o * rank;
    for (let r = 0; r < rank; r++) sum += up[offset + r] * lowRank[r];
    result[o] = sum * scale;
  }
  return { lowRank, result };
}

button.addEventListener("click", async () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    setStatus("Choose the Personal Ami .safetensors file in the local inspector above first.", "error");
    return;
  }

  button.disabled = true;
  summary.replaceChildren();
  summary.hidden = false;
  output.textContent = "";
  output.hidden = false;

  try {
    setStatus("Reading the local safetensors header…");
    const started = performance.now();
    const { header, dataStart } = await readSafetensorsHeader(file);
    const manifestResponse = await fetch(MANIFEST_URL, { credentials: "omit" });
    if (!manifestResponse.ok) throw new Error("Model manifest request failed: HTTP " + manifestResponse.status);
    const manifest = await manifestResponse.json();
    const indexPath = manifest.components?.text_encoder?.weightsIndex;
    if (!indexPath) throw new Error("Model manifest has no CLIP-L weights index.");
    const indexResponse = await fetch(MODEL_BASE + indexPath, { credentials: "omit" });
    if (!indexResponse.ok) throw new Error("CLIP-L index request failed: HTTP " + indexResponse.status);
    const indexBuffer = await indexResponse.arrayBuffer();
    const fileRecord = (manifest.files || []).find(item => item.path === indexPath);
    if (fileRecord?.size !== undefined && indexBuffer.byteLength !== fileRecord.size) {
      throw new Error("CLIP-L weight-index JSON size does not match the manifest.");
    }
    if (fileRecord?.sha256) {
      const digest = await crypto.subtle.digest("SHA-256", indexBuffer);
      const actual = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, "0")).join("");
      if (actual !== fileRecord.sha256) throw new Error("CLIP-L weight-index SHA-256 mismatch.");
    }
    const index = JSON.parse(new TextDecoder().decode(indexBuffer));
    const match = findRepresentativeLinearPair(header, index);
    const { stem, pair, alphaName, alphaInfo, target } = match;

    setStatus("Reading one representative local BF16 down/up pair and its scalar alpha…");
    const down = await readTensorPayload(file, dataStart, pair.down, pair.downName);
    const up = await readTensorPayload(file, dataStart, pair.up, pair.upName);
    const alphaValues = await readTensorPayload(file, dataStart, alphaInfo, alphaName);
    if (alphaValues.length !== 1) throw new Error("Expected one scalar alpha value.");
    const rank = pair.down.shape[0];
    const alpha = alphaValues[0];
    const scale = alpha / rank;
    const inputSize = pair.down.shape[1];
    const outputSize = pair.up.shape[0];

    const input = new Float32Array(inputSize);
    for (let i = 0; i < input.length; i++) input[i] = Math.sin((i + 1) * 0.017) * 0.5;
    const result = applyLinearLora(down, up, input, rank, outputSize, scale);

    let maxAbs = 0;
    let sumSquares = 0;
    for (let i = 0; i < result.result.length; i++) {
      const value = result.result[i];
      if (!Number.isFinite(value)) throw new Error("Computed LoRA output contains a non-finite value at " + i + ".");
      maxAbs = Math.max(maxAbs, Math.abs(value));
      sumSquares += value * value;
    }
    if (!(maxAbs > 0)) throw new Error("The selected LoRA pair produced an all-zero delta on the deterministic test vector.");

    addRow("Local adapter", file.name + " · " + (file.size / 1024 / 1024).toFixed(1) + " MiB");
    addRow("Component", "CLIP-L text encoder");
    addRow("LoRA target", stem);
    addRow("Mapped base weight", target.moduleName + " → " + target.entry.initializer);
    addRow("Target Torch shape", "[" + target.entry.torchShape.join(", ") + "]");
    addRow("Adapter down/up", "[" + pair.down.shape.join(", ") + "] / [" + pair.up.shape.join(", ") + "]");
    addRow("ONNX transposed flag", String(!!target.entry.transposed));
    addRow("Rank / alpha / scale", rank + " / " + alpha.toFixed(4) + " / " + scale.toFixed(6));
    addRow("Local payload bytes read", (down.byteLength + up.byteLength + alphaValues.byteLength).toLocaleString());
    addRow("Synthetic input/output", input.length.toLocaleString() + " → " + result.result.length.toLocaleString() + " FP32 values");
    addRow("Computed delta max |value|", maxAbs.toExponential(4));
    addRow("Computed delta L2 norm", Math.sqrt(sumSquares).toExponential(4));
    addRow("Elapsed time", ((performance.now() - started) / 1000).toFixed(2) + " s");

    output.textContent = [
      "The selected local BF16 LoRA down/up payloads were decoded and applied as:",
      "delta(x) = (alpha / rank) × W_up × (W_down × x)",
      "",
      "This operation ran in the browser using only the selected .safetensors byte ranges.",
      "No LoRA tensor bytes were uploaded or sent to a server.",
      "The target was checked against the CLIP-L FP16 weights index using the original Torch shape.",
      "",
      "This does not yet patch the ONNX external-data shard or validate model output equivalence."
    ].join("\n");

    setStatus("One real local Personal Ami BF16 CLIP-L pair decoded and produced a finite, nonzero low-rank delta. No model weights or LoRA payload were uploaded.", "success");
  } catch (error) {
    console.error("[Illustrious] LoRA payload smoke test failed:", error);
    setStatus("LoRA payload smoke test failed: " + (error?.message || String(error)), "error");
  } finally {
    button.disabled = false;
  }
});
