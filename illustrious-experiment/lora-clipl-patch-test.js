import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";
import { acquireHeavyModelTest, releaseHeavyModelTest } from "./heavy-model-test-lock.js";

const MODEL_ID = "aaaaaeeeee/blip-sdxl-anime-wai-illustrious17";
const MODEL_BASE = "https://huggingface.co/" + MODEL_ID + "/resolve/main/";
const MANIFEST_URL = MODEL_BASE + "manifest.json";
const HEADER_LIMIT = 64 * 1024 * 1024;

ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

const fileInput = document.querySelector("#loraFile");
const button = document.querySelector("#testSingleLoraPatch");
const strengthInput = document.querySelector("#loraPatchStrength");
const status = document.querySelector("#loraPatchStatus");
const summary = document.querySelector("#loraPatchSummary");
const output = document.querySelector("#loraPatchOutput");

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

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "Unknown";
  if (bytes >= 1024 ** 3) return (bytes / 1024 ** 3).toFixed(2) + " GiB";
  if (bytes >= 1024 ** 2) return (bytes / 1024 ** 2).toFixed(1) + " MiB";
  return (bytes / 1024).toFixed(1) + " KiB";
}

async function readSafetensorsHeader(file) {
  if (!file || file.size < 8) throw new Error("Choose the Personal Ami .safetensors file first.");
  const prefix = await file.slice(0, 8).arrayBuffer();
  const headerLength = new DataView(prefix).getBigUint64(0, true);
  if (headerLength === 0n || headerLength > BigInt(HEADER_LIMIT)) {
    throw new Error("Safetensors header is invalid or exceeds the 64 MiB limit.");
  }
  if (headerLength > BigInt(file.size - 8)) throw new Error("Safetensors file ends before its declared header.");
  const dataStart = 8 + Number(headerLength);
  const bytes = await file.slice(8, dataStart).arrayBuffer();
  const header = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  if (!header || typeof header !== "object" || Array.isArray(header)) {
    throw new Error("Safetensors header must be a JSON object.");
  }
  return { header, dataStart };
}

function findRepresentativeLinearPair(header, index) {
  if (!index?.entries || typeof index.entries !== "object") {
    throw new Error("CLIP-L weights index has no entries object.");
  }

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

  throw new Error("No shape-compatible BF16 CLIP-L linear pair with a scalar alpha entry was found.");
}

function readBf16(buffer, label) {
  if (buffer.byteLength % 2) throw new Error(label + " has an odd BF16 byte count.");
  const view = new DataView(buffer);
  const bits = new Uint32Array(buffer.byteLength / 2);
  for (let i = 0; i < bits.length; i++) {
    const bf16 = view.getUint16(i * 2, true);
    const exponent = bf16 & 0x7f80;
    if (exponent === 0x7f80) throw new Error(label + " contains BF16 NaN/Infinity at value " + i + ".");
    bits[i] = bf16 << 16;
  }
  return new Float32Array(bits.buffer);
}

async function readLocalBf16(file, dataStart, info, label) {
  if (!info || info.dtype !== "BF16" || !Array.isArray(info.shape) ||
      !Array.isArray(info.data_offsets) || info.data_offsets.length !== 2) {
    throw new Error(label + " is not a valid BF16 safetensors record.");
  }
  const [start, end] = info.data_offsets;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) ||
      start < 0 || end < start || dataStart + end > file.size) {
    throw new Error(label + " has an invalid local byte range.");
  }
  const elements = info.shape.reduce((n, dim) => n * dim, 1);
  if (!Number.isSafeInteger(elements) || (end - start) !== elements * 2) {
    throw new Error(label + " byte count does not match its BF16 shape.");
  }
  const buffer = await file.slice(dataStart + start, dataStart + end).arrayBuffer();
  return readBf16(buffer, label);
}

async function fetchVerifiedJson(path, fileMap) {
  const expected = fileMap.get(path);
  if (!expected) throw new Error("Model manifest has no record for " + path + ".");
  const response = await fetch(MODEL_BASE + path, { credentials: "omit", cache: "force-cache" });
  if (!response.ok) throw new Error(path + " request failed: HTTP " + response.status);
  const buffer = await response.arrayBuffer();
  if (expected.size !== undefined && buffer.byteLength !== expected.size) {
    throw new Error(path + " size mismatch: expected " + expected.size + ", received " + buffer.byteLength + ".");
  }
  if (expected.sha256) {
    const digest = await crypto.subtle.digest("SHA-256", buffer);
    const actual = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, "0")).join("");
    if (actual !== expected.sha256) throw new Error(path + " SHA-256 mismatch.");
  }
  return JSON.parse(new TextDecoder().decode(buffer));
}

function resolveShardPath(entry, component) {
  const shard = entry?.shard;
  if (typeof shard === "number" && Number.isInteger(shard) &&
      shard >= 0 && shard < component.shards.length) {
    return component.shards[shard];
  }
  if (typeof shard === "string" && shard.length) {
    const normalized = shard.replaceAll("\\", "/");
    const exact = component.shards.find(path => path === normalized);
    if (exact) return exact;
    const basename = normalized.slice(normalized.lastIndexOf("/") + 1);
    const matches = component.shards.filter(path =>
      path === basename || path.endsWith("/" + basename)
    );
    if (matches.length === 1) return matches[0];
  }
  if (component.shards.length === 1 && shard == null) return component.shards[0];
  throw new Error("Could not resolve index shard " + JSON.stringify(shard) +
    " against component shards: " + component.shards.join(", "));
}

function patchLinearTarget(baseBuffer, down, up, targetShape, transposed, scale) {
  const [outFeatures, inFeatures] = targetShape;
  const rank = down.length / inFeatures;
  if (!Number.isInteger(rank) || up.length !== outFeatures * rank) {
    throw new Error("LoRA factor sizes do not match the target matrix dimensions.");
  }
  if (baseBuffer.byteLength !== outFeatures * inFeatures * 2) {
    throw new Error("Target weight byte length does not match its Torch shape.");
  }
  if (typeof Float16Array !== "function") {
    throw new Error("This browser does not support Float16Array.");
  }

  const base = new Float16Array(baseBuffer);
  const patched = new Float16Array(base.length);
  const rowDelta = new Float32Array(inFeatures);
  let maxAbsDelta = 0;
  let maxAbsApplied = 0;

  for (let out = 0; out < outFeatures; out++) {
    rowDelta.fill(0);
    const upOffset = out * rank;
    for (let r = 0; r < rank; r++) {
      const coefficient = up[upOffset + r];
      const downOffset = r * inFeatures;
      for (let input = 0; input < inFeatures; input++) {
        rowDelta[input] += coefficient * down[downOffset + input];
      }
    }
    for (let input = 0; input < inFeatures; input++) {
      const delta = rowDelta[input] * scale;
      const baseIndex = transposed ? input * outFeatures + out : out * inFeatures + input;
      const next = Number(base[baseIndex]) + delta;
      if (!Number.isFinite(next)) {
        throw new Error("Patched FP16 target would become non-finite at [" + out + "," + input + "].");
      }
      patched[baseIndex] = next;
      maxAbsDelta = Math.max(maxAbsDelta, Math.abs(rowDelta[input]));
      maxAbsApplied = Math.max(maxAbsApplied, Math.abs(delta));
    }
  }

  if (!(maxAbsApplied > 0)) throw new Error("The selected adapter pair produced an all-zero target delta.");
  return {
    bytes: new Uint8Array(patched.buffer.slice(patched.byteOffset, patched.byteOffset + patched.byteLength)),
    maxAbsDelta,
    maxAbsApplied
  };
}

function readFp16Output(tensor, label) {
  if (!tensor?.data || tensor.type !== "float16" || !Array.isArray(tensor.dims)) {
    throw new Error(label + " is not a readable FP16 output.");
  }
  let values;
  if (tensor.data instanceof Float16Array) values = tensor.data;
  else if (tensor.data instanceof Uint16Array) {
    values = new Float16Array(tensor.data.buffer.slice(
      tensor.data.byteOffset, tensor.data.byteOffset + tensor.data.byteLength
    ));
  } else {
    throw new Error(label + " has unsupported backing type " + tensor.data.constructor.name + ".");
  }
  const copy = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    copy[i] = Number(values[i]);
    if (!Number.isFinite(copy[i])) throw new Error(label + " contains non-finite output at " + i + ".");
  }
  return { values: copy, dims: tensor.dims.slice() };
}

function makeTestIds() {
  const ids = new BigInt64Array(77);
  ids.fill(49407n);
  ids[0] = 49406n;
  ids[1] = 1000n;
  ids[2] = 49407n;
  return ids;
}

function compareOutputs(baseline, patched) {
  if (baseline.dims.length !== patched.dims.length ||
      baseline.dims.some((value, i) => value !== patched.dims[i]) ||
      baseline.values.length !== patched.values.length) {
    throw new Error("Baseline and patched CLIP-L output shapes differ.");
  }
  let maxAbs = 0;
  let sumSquares = 0;
  let changed = 0;
  for (let i = 0; i < baseline.values.length; i++) {
    const delta = Math.abs(patched.values[i] - baseline.values[i]);
    maxAbs = Math.max(maxAbs, delta);
    sumSquares += delta * delta;
    if (delta > 0) changed++;
  }
  return {
    maxAbs,
    rms: Math.sqrt(sumSquares / baseline.values.length),
    changed,
    count: baseline.values.length
  };
}

async function runClip(session, label, ids) {
  if (!session.inputNames.includes("input_ids")) {
    throw new Error(label + " graph is missing input_ids.");
  }
  const results = await session.run({
    input_ids: new ort.Tensor("int64", ids, [1, 77])
  });
  const output = results.penultimate_hidden_state;
  if (!output) throw new Error(label + " did not return penultimate_hidden_state.");
  return readFp16Output(output, label + " penultimate_hidden_state");
}

async function loadManifest() {
  const response = await fetch(MANIFEST_URL, { credentials: "omit", cache: "force-cache" });
  if (!response.ok) throw new Error("Model manifest request failed: HTTP " + response.status);
  return response.json();
}

button.addEventListener("click", async () => {
  const file = fileInput.files && fileInput.files[0];
  const strength = Number(strengthInput.value);
  if (!file) {
    setStatus("Choose Personal Ami in the local .safetensors picker above first.", "error");
    return;
  }
  if (!Number.isFinite(strength) || strength < 0 || strength > 2) {
    setStatus("Strength must be between 0 and 2.", "error");
    return;
  }

  button.disabled = true;
  summary.replaceChildren();
  summary.hidden = false;
  output.textContent = "";
  output.hidden = false;
  let lockAcquired = false;
  let baselineSession = null;
  let patchedSession = null;

  try {
    acquireHeavyModelTest("single-pair CLIP-L LoRA patch validation");
    lockAcquired = true;
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("The browser did not return a WebGPU adapter.");

    setStatus("Reading the local safetensors header and fetching the model manifest…");
    const started = performance.now();
    const { header, dataStart } = await readSafetensorsHeader(file);
    const manifest = await loadManifest();
    const fileMap = new Map((manifest.files || []).map(item => [item.path, item]));
    const component = manifest.components?.text_encoder;
    if (!component?.onnx || !component?.weightsIndex ||
        !Array.isArray(component.shards) || component.shards.length !== 1) {
      throw new Error("The candidate CLIP-L component does not match the expected single-shard layout.");
    }
    const index = await fetchVerifiedJson(component.weightsIndex, fileMap);
    const match = findRepresentativeLinearPair(header, index);
    const { stem, pair, alphaName, alphaInfo, target } = match;
    const entry = target.entry;
    const [outFeatures, inFeatures] = entry.torchShape;

    for (const key of ["initializer", "offset", "length"]) {
      if (entry[key] === undefined || entry[key] === null) {
        throw new Error("Mapped index entry is missing required field " + key + ".");
      }
    }
    const offset = Number(entry.offset);
    const length = Number(entry.length);
    if (!Number.isSafeInteger(offset) || offset < 0 ||
        !Number.isSafeInteger(length) || length !== outFeatures * inFeatures * 2) {
      throw new Error("Mapped target offset/length is invalid for the declared FP16 matrix.");
    }

    const shardPath = resolveShardPath(entry, component);
    const shardRecord = fileMap.get(shardPath);
    if (!shardRecord || shardRecord.size === undefined ||
        offset + length > shardRecord.size) {
      throw new Error("Mapped target byte range extends beyond shard " + shardPath + ".");
    }

    setStatus("Decoding the selected local BF16 LoRA pair and computing its FP16 delta…");
    const [down, up, alphaValues] = await Promise.all([
      readLocalBf16(file, dataStart, pair.down, pair.downName),
      readLocalBf16(file, dataStart, pair.up, pair.upName),
      readLocalBf16(file, dataStart, alphaInfo, alphaName)
    ]);
    if (alphaValues.length !== 1) throw new Error("LoRA alpha tensor must contain exactly one scalar.");
    const rank = pair.down.shape[0];
    const scale = (alphaValues[0] / rank) * strength;
    if (!Number.isFinite(scale)) throw new Error("LoRA alpha/rank strength is non-finite.");

    setStatus("Fetching the base CLIP-L external-data shard…");
    const shardResponse = await fetch(MODEL_BASE + shardPath, {
      credentials: "omit",
      cache: "force-cache"
    });
    if (!shardResponse.ok) throw new Error("CLIP-L shard request failed: HTTP " + shardResponse.status);
    const baseBlob = await shardResponse.blob();
    if (baseBlob.size !== shardRecord.size) {
      throw new Error("CLIP-L shard size mismatch: expected " + shardRecord.size +
        ", received " + baseBlob.size + ".");
    }
    let shardHashStatus = "not listed in manifest";
    if (shardRecord.sha256) {
      const shardBuffer = await baseBlob.arrayBuffer();
      const digest = await crypto.subtle.digest("SHA-256", shardBuffer);
      const actualHash = Array.from(new Uint8Array(digest),
        value => value.toString(16).padStart(2, "0")).join("");
      if (actualHash !== shardRecord.sha256) {
        throw new Error("CLIP-L shard SHA-256 does not match the model manifest.");
      }
      shardHashStatus = "matched manifest SHA-256";
    }

    const baseTargetBuffer = await baseBlob.slice(offset, offset + length).arrayBuffer();
    const patch = patchLinearTarget(
      baseTargetBuffer,
      down,
      up,
      entry.torchShape,
      !!entry.transposed,
      scale
    );
    const patchedBlob = new Blob([
      baseBlob.slice(0, offset),
      patch.bytes,
      baseBlob.slice(offset + length)
    ], { type: "application/octet-stream" });

    addRow("Local adapter", file.name + " · " + formatBytes(file.size));
    addRow("Mapped target", target.moduleName + " → " + entry.initializer);
    addRow("Target Torch shape", "[" + entry.torchShape.join(", ") + "]");
    addRow("LoRA down/up shapes", "[" + pair.down.shape.join(", ") + "] / [" + pair.up.shape.join(", ") + "]");
    addRow("Target shard", shardPath + " · byte offset " + offset + " · " + formatBytes(length));
    addRow("ONNX transposed flag", String(!!entry.transposed));
    addRow("Rank / alpha", rank + " / " + alphaValues[0].toFixed(5));
    addRow("Applied strength", strength.toFixed(3) + " · effective alpha/rank " + scale.toFixed(6));
    addRow("Maximum |raw LoRA delta|", patch.maxAbsDelta.toExponential(4));
    addRow("Maximum |applied delta|", patch.maxAbsApplied.toExponential(4));
    addRow("Patched shard representation", "Blob composed from original shard slices + one FP16 weight update");
    addRow("Full base shard download", formatBytes(baseBlob.size));
    addRow("Base shard integrity", shardHashStatus);

    const ids = makeTestIds();
    setStatus("Loading the original CLIP-L session and measuring its baseline output…");
    baselineSession = await ort.InferenceSession.create(MODEL_BASE + component.onnx, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "basic",
      externalData: component.shards.map(path => ({
        path: path.slice(path.lastIndexOf("/") + 1),
        data: MODEL_BASE + path
      }))
    });
    const baseline = await runClip(baselineSession, "Baseline CLIP-L", ids);
    await baselineSession.release();
    baselineSession = null;

    setStatus("Loading CLIP-L again with the patched shard supplied as a Blob…");
    patchedSession = await ort.InferenceSession.create(MODEL_BASE + component.onnx, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "basic",
      externalData: component.shards.map(path => ({
        path: path.slice(path.lastIndexOf("/") + 1),
        data: path === shardPath ? patchedBlob : MODEL_BASE + path
      }))
    });
    const patched = await runClip(patchedSession, "Patched CLIP-L", ids);
    const difference = compareOutputs(baseline, patched);

    addRow("Baseline / patched output", "[" + baseline.dims.join(", ") + "] / [" + patched.dims.join(", ") + "]");
    addRow("Output values changed", difference.changed.toLocaleString() + " / " + difference.count.toLocaleString());
    addRow("Max |output difference|", difference.maxAbs.toExponential(5));
    addRow("Output RMS difference", difference.rms.toExponential(5));
    addRow("Patch + inference elapsed", ((performance.now() - started) / 1000).toFixed(1) + " s");

    if (!difference.changed || !(difference.maxAbs > 0)) {
      throw new Error("The patched target did not measurably change CLIP-L output. Check the index's storage orientation and mapped byte range.");
    }

    output.textContent = [
      "One real Personal Ami BF16 LoRA pair was fused into one CLIP-L FP16 ONNX initializer.",
      "The graph and its external-data shard were not modified on the server or in the repository.",
      "ONNX Runtime loaded the patched external-data Blob and completed inference.",
      "The same input IDs produced a measurable output change relative to the unmodified base.",
      "",
      "This validates one linear target only. It does not yet apply the full 980-pair adapter or prove Diffusers numerical equivalence."
    ].join("\n");
    setStatus("Single-pair LoRA patch passed: the original and patched CLIP-L sessions both ran, and the same input produced a measurable output difference.", "success");
  } catch (error) {
    console.error("[Illustrious] Single-pair LoRA patch test failed:", error);
    setStatus("Single-pair LoRA patch test failed: " + (error?.message || String(error)), "error");
    output.textContent = error?.stack || error?.message || String(error);
  } finally {
    if (patchedSession) {
      try { await patchedSession.release(); } catch (error) {
        console.warn("[Illustrious] Could not release patched CLIP-L session:", error);
      }
    }
    if (baselineSession) {
      try { await baselineSession.release(); } catch (error) {
        console.warn("[Illustrious] Could not release baseline CLIP-L session:", error);
      }
    }
    if (lockAcquired) releaseHeavyModelTest("single-pair CLIP-L LoRA patch validation");
    button.disabled = false;
  }
});
