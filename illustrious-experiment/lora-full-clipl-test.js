import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";
import { acquireHeavyModelTest, releaseHeavyModelTest } from "./heavy-model-test-lock.js";

const MODEL_BASE = "https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/resolve/main/";
const fileInput = document.querySelector("#loraFile");
const button = document.querySelector("#testFullClipL");
const strengthInput = document.querySelector("#fullClipLStrength");
const status = document.querySelector("#fullClipLStatus");
const summary = document.querySelector("#fullClipLSummary");
const output = document.querySelector("#fullClipLOutput");

ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

function setStatus(message, type = "info") {
  status.textContent = message;
  status.dataset.type = type;
}

function row(label, value) {
  const item = document.createElement("div");
  item.className = "data-row";
  const left = document.createElement("span");
  left.className = "data-label";
  left.textContent = label;
  const right = document.createElement("span");
  right.className = "data-value";
  right.textContent = value;
  item.append(left, right);
  summary.append(item);
}

async function readHeader(file) {
  const prefix = await file.slice(0, 8).arrayBuffer();
  const length = new DataView(prefix).getBigUint64(0, true);
  if (!length || length > 64n * 1024n * 1024n || length > BigInt(file.size - 8)) {
    throw new Error("Invalid or oversized safetensors header.");
  }
  const dataStart = 8 + Number(length);
  const header = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(
    await file.slice(8, dataStart).arrayBuffer()
  ));
  return { header, dataStart };
}

function readBf16(buffer, label) {
  if (buffer.byteLength % 2) throw new Error(label + " has odd BF16 byte length.");
  const view = new DataView(buffer);
  const bits = new Uint32Array(buffer.byteLength / 2);
  for (let i = 0; i < bits.length; i++) {
    const value = view.getUint16(i * 2, true);
    if ((value & 0x7f80) === 0x7f80) throw new Error(label + " contains NaN/Infinity.");
    bits[i] = value << 16;
  }
  return new Float32Array(bits.buffer);
}

async function readTensor(file, dataStart, name, info) {
  if (!info || info.dtype !== "BF16" || !Array.isArray(info.shape) ||
      !Array.isArray(info.data_offsets) || info.data_offsets.length !== 2) {
    throw new Error("Missing or unsupported BF16 tensor " + name + ".");
  }
  const [start, end] = info.data_offsets;
  const count = info.shape.reduce((a, b) => a * b, 1);
  if (!Number.isSafeInteger(count) || end - start !== count * 2 ||
      start < 0 || end < start || dataStart + end > file.size) {
    throw new Error("Invalid byte range for " + name + ".");
  }
  return readBf16(await file.slice(dataStart + start, dataStart + end).arrayBuffer(), name);
}

function resolveShard(entry, component) {
  const value = entry.shard;
  if (typeof value === "number" && Number.isInteger(value) &&
      value >= 0 && value < component.shards.length) return component.shards[value];
  if (typeof value === "string") {
    const normalized = value.replaceAll("\\", "/");
    const exact = component.shards.find(path => path === normalized);
    if (exact) return exact;
    const basename = normalized.split("/").pop();
    const matches = component.shards.filter(path => path === basename || path.endsWith("/" + basename));
    if (matches.length === 1) return matches[0];
  }
  if (component.shards.length === 1 && value == null) return component.shards[0];
  throw new Error("Cannot resolve shard " + JSON.stringify(value) + ".");
}

function patchLinear(buffer, down, up, shape, transposed, scale, label) {
  const [outFeatures, inFeatures] = shape;
  const rank = down.length / inFeatures;
  if (!Number.isInteger(rank) || up.length !== outFeatures * rank ||
      buffer.byteLength !== outFeatures * inFeatures * 2) {
    throw new Error("Shape mismatch while patching " + label + ".");
  }
  const base = new Float16Array(buffer);
  const patched = new Float16Array(base.length);
  const deltaRow = new Float32Array(inFeatures);
  for (let out = 0; out < outFeatures; out++) {
    deltaRow.fill(0);
    for (let r = 0; r < rank; r++) {
      const coefficient = up[out * rank + r];
      const offset = r * inFeatures;
      for (let input = 0; input < inFeatures; input++) {
        deltaRow[input] += coefficient * down[offset + input];
      }
    }
    for (let input = 0; input < inFeatures; input++) {
      const index = transposed ? input * outFeatures + out : out * inFeatures + input;
      const value = Number(base[index]) + deltaRow[input] * scale;
      if (!Number.isFinite(value)) throw new Error("Non-finite patched weight in " + label + ".");
      patched[index] = value;
    }
  }
  return new Uint8Array(patched.buffer);
}

async function verifiedJson(path, fileMap) {
  const record = fileMap.get(path);
  if (!record) throw new Error("Manifest is missing " + path + ".");
  const response = await fetch(MODEL_BASE + path, { credentials: "omit" });
  if (!response.ok) throw new Error(path + " returned HTTP " + response.status + ".");
  const buffer = await response.arrayBuffer();
  if (record.size !== undefined && buffer.byteLength !== record.size) throw new Error(path + " size mismatch.");
  if (record.sha256) {
    const hash = await crypto.subtle.digest("SHA-256", buffer);
    const actual = Array.from(new Uint8Array(hash), v => v.toString(16).padStart(2, "0")).join("");
    if (actual !== record.sha256) throw new Error(path + " SHA-256 mismatch.");
  }
  return JSON.parse(new TextDecoder().decode(buffer));
}

function compare(a, b) {
  if (a.dims.join(",") !== b.dims.join(",")) throw new Error("Baseline and patched output shapes differ.");
  let changed = 0, max = 0, squares = 0;
  for (let i = 0; i < a.values.length; i++) {
    const delta = Math.abs(a.values[i] - b.values[i]);
    if (delta) changed++;
    max = Math.max(max, delta);
    squares += delta * delta;
  }
  return { changed, max, rms: Math.sqrt(squares / a.values.length), count: a.values.length };
}

async function runClip(session) {
  const ids = new BigInt64Array(77);
  ids.fill(49407n);
  ids[0] = 49406n;
  ids[1] = 1000n;
  ids[2] = 49407n;
  const result = await session.run({ input_ids: new ort.Tensor("int64", ids, [1, 77]) });
  const tensor = result.penultimate_hidden_state;
  if (!tensor || tensor.type !== "float16") throw new Error("CLIP-L output is missing or not FP16.");
  const source = tensor.data instanceof Float16Array ? tensor.data :
    new Float16Array(tensor.data.buffer.slice(tensor.data.byteOffset, tensor.data.byteOffset + tensor.data.byteLength));
  const values = new Float32Array(source.length);
  for (let i = 0; i < source.length; i++) {
    values[i] = Number(source[i]);
    if (!Number.isFinite(values[i])) throw new Error("Non-finite CLIP-L output at " + i + ".");
  }
  return { values, dims: tensor.dims.slice() };
}

button.addEventListener("click", async () => {
  const file = fileInput.files?.[0];
  const strength = Number(strengthInput.value);
  if (!file) return setStatus("Select Personal Ami in the shared .safetensors picker first.", "error");
  if (!Number.isFinite(strength) || strength < 0 || strength > 2) {
    return setStatus("Strength must be between 0 and 2.", "error");
  }
  button.disabled = true;
  summary.replaceChildren();
  summary.hidden = false;
  output.textContent = "";
  output.hidden = false;
  let lock = false, baselineSession = null, patchedSession = null;
  try {
    acquireHeavyModelTest("full CLIP-L Personal Ami patch test");
    lock = true;
    setStatus("Reading the local adapter header and verifying the CLIP-L index…");
    const started = performance.now();
    const { header, dataStart } = await readHeader(file);
    const manifestResponse = await fetch(MODEL_BASE + "manifest.json", { credentials: "omit" });
    if (!manifestResponse.ok) throw new Error("Model manifest request failed.");
    const manifest = await manifestResponse.json();
    const fileMap = new Map((manifest.files || []).map(item => [item.path, item]));
    const component = manifest.components?.text_encoder;
    if (!component?.onnx || !component.weightsIndex || !Array.isArray(component.shards)) {
      throw new Error("CLIP-L manifest component is incomplete.");
    }
    const index = await verifiedJson(component.weightsIndex, fileMap);
    const lookup = new Map();
    for (const [name, entry] of Object.entries(index.entries || {})) {
      const key = (name.endsWith(".weight") ? name.slice(0, -7) : name).replaceAll(".", "_");
      lookup.set(key, { name, entry });
    }

    const pairs = new Map();
    for (const [name, info] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      let side = null, stem = name;
      if (name.startsWith("lora_te1_") && name.endsWith(".lora_down.weight")) {
        side = "down"; stem = name.slice(0, -".lora_down.weight".length);
      } else if (name.startsWith("lora_te1_") && name.endsWith(".lora_up.weight")) {
        side = "up"; stem = name.slice(0, -".lora_up.weight".length);
      }
      if (side) {
        if (!pairs.has(stem)) pairs.set(stem, {});
        pairs.get(stem)[side] = info;
        pairs.get(stem)[side + "Name"] = name;
      }
    }

    const patchesByShard = new Map();
    let mapped = 0, skipped = 0, payloadBytes = 0;
    for (const [stem, pair] of pairs) {
      if (!pair.down || !pair.up) throw new Error("Incomplete CLIP-L pair " + stem + ".");
      const key = stem.slice("lora_te1_".length);
      const target = lookup.get(key);
      if (!target) {
        if (/^text_model_encoder_layers_11_/.test(key)) { skipped++; continue; }
        throw new Error("No CLIP-L index entry for " + stem + ".");
      }
      const entry = target.entry;
      if (!Array.isArray(entry.torchShape) || entry.torchShape.length !== 2) {
        throw new Error("Unsupported non-linear CLIP-L target " + target.name + ".");
      }
      for (const field of ["offset", "length", "initializer"]) {
        if (entry[field] === undefined || entry[field] === null) throw new Error("Index entry missing " + field + " for " + target.name + ".");
      }
      const shard = resolveShard(entry, component);
      const record = fileMap.get(shard);
      const offset = Number(entry.offset), length = Number(entry.length);
      const [outFeatures, inFeatures] = entry.torchShape;
      if (!record || !Number.isSafeInteger(offset) || offset < 0 ||
          !Number.isSafeInteger(length) || length !== outFeatures * inFeatures * 2 ||
          offset + length > record.size) {
        throw new Error("Invalid indexed byte range for " + target.name + ".");
      }
      const down = await readTensor(file, dataStart, pair.downName, pair.down);
      const up = await readTensor(file, dataStart, pair.upName, pair.up);
      const alphaName = stem + ".alpha";
      const alphaInfo = header[alphaName];
      const alpha = await readTensor(file, dataStart, alphaName, alphaInfo);
      if (alpha.length !== 1) throw new Error("Alpha must be scalar for " + stem + ".");
      const rank = pair.down.shape[0];
      if (pair.down.shape.length !== 2 || pair.up.shape.length !== 2 ||
          pair.down.shape[1] !== inFeatures || pair.up.shape[0] !== outFeatures ||
          pair.down.shape[0] !== pair.up.shape[1]) {
        throw new Error("LoRA dimensions do not match target " + target.name + ".");
      }
      const scale = alpha[0] / rank * strength;
      const group = patchesByShard.get(shard) || [];
      group.push({ offset, length, target: target.name, apply: async base => {
        const delta = patchLinear(base, down, up, entry.torchShape, !!entry.transposed, scale, target.name);
        return delta;
      }});
      patchesByShard.set(shard, group);
      mapped++;
      payloadBytes += (pair.down.data_offsets[1] - pair.down.data_offsets[0]) +
        (pair.up.data_offsets[1] - pair.up.data_offsets[0]);
    }

    if (mapped !== 66 || skipped !== 6) {
      throw new Error("Expected 66 mapped CLIP-L pairs + 6 unindexed final-block pairs; found " + mapped + " mapped and " + skipped + " skipped.");
    }
    setStatus("Mapped all 66 indexed CLIP-L pairs. Downloading and verifying the base shard…");
    const patchedExternalData = [];
    for (const shard of component.shards) {
      const record = fileMap.get(shard);
      if (!record) throw new Error("Manifest has no shard record for " + shard + ".");
      const response = await fetch(MODEL_BASE + shard, { credentials: "omit" });
      if (!response.ok) throw new Error("Shard download failed: " + shard);
      const base = await response.blob();
      if (base.size !== record.size) throw new Error("Shard size mismatch for " + shard + ".");
      if (record.sha256) {
        const digest = await crypto.subtle.digest("SHA-256", await base.arrayBuffer());
        const actual = Array.from(new Uint8Array(digest), v => v.toString(16).padStart(2, "0")).join("");
        if (actual !== record.sha256) throw new Error("Shard SHA-256 mismatch for " + shard + ".");
      }
      const patches = (patchesByShard.get(shard) || []).sort((a, b) => a.offset - b.offset);
      const parts = [];
      let cursor = 0;
      for (const patch of patches) {
        if (patch.offset < cursor) throw new Error("Overlapping ONNX patch ranges near " + patch.target + ".");
        parts.push(base.slice(cursor, patch.offset));
        const bytes = await patch.apply(await base.slice(patch.offset, patch.offset + patch.length).arrayBuffer());
        parts.push(bytes);
        cursor = patch.offset + patch.length;
      }
      parts.push(base.slice(cursor));
      const blob = patches.length ? new Blob(parts, { type: "application/octet-stream" }) : base;
      patchedExternalData.push({ path: shard.slice(shard.lastIndexOf("/") + 1), data: blob });
    }

    row("Adapter", file.name);
    row("Applied strength", strength.toFixed(3));
    row("Mapped CLIP-L targets", mapped + " / 66");
    row("Unindexed final layer-11 pairs", String(skipped));
    row("LoRA factor payload read", (payloadBytes / 1024 / 1024).toFixed(2) + " MiB");
    row("Patched shards", String(patchedExternalData.filter(x => x.data instanceof Blob).length));
    row("Patch preparation", ((performance.now() - started) / 1000).toFixed(1) + " s");

    const ids = new BigInt64Array(77);
    ids.fill(49407n); ids[0] = 49406n; ids[1] = 1000n; ids[2] = 49407n;
    setStatus("Loading original CLIP-L and measuring baseline output…");
    baselineSession = await ort.InferenceSession.create(MODEL_BASE + component.onnx, {
      executionProviders: ["webgpu"], graphOptimizationLevel: "basic",
      externalData: component.shards.map(path => ({ path: path.split("/").pop(), data: MODEL_BASE + path }))
    });
    const baseline = await runClip(baselineSession);
    await baselineSession.release(); baselineSession = null;

    setStatus("Loading CLIP-L with all 66 Personal Ami patches…");
    patchedSession = await ort.InferenceSession.create(MODEL_BASE + component.onnx, {
      executionProviders: ["webgpu"], graphOptimizationLevel: "basic",
      externalData: patchedExternalData
    });
    const patched = await runClip(patchedSession);
    const difference = compare(baseline, patched);
    row("Changed output values", difference.changed.toLocaleString() + " / " + difference.count.toLocaleString());
    row("Maximum output difference", difference.max.toExponential(5));
    row("Output RMS difference", difference.rms.toExponential(5));
    row("Total test time", ((performance.now() - started) / 1000).toFixed(1) + " s");
    if (!difference.changed || !Number.isFinite(difference.rms)) throw new Error("Full CLIP-L patch produced no finite output difference.");
    output.textContent = "All 66 indexed Personal Ami CLIP-L pairs were fused into their FP16 ONNX initializers. The six unindexed final-layer pairs were excluded because the graph exposes penultimate_hidden_state. The original and patched CLIP-L sessions both ran successfully on identical inputs. This validates the complete indexed CLIP-L adapter component only; CLIP-bigG and all UNet partitions remain untested.";
    setStatus("Full CLIP-L adapter patch passed: all 66 mapped pairs were applied and ONNX Runtime inference succeeded.", "success");
  } catch (error) {
    console.error("[Illustrious] Full CLIP-L LoRA test failed:", error);
    setStatus("Full CLIP-L patch failed: " + (error?.message || String(error)), "error");
    output.textContent = error?.stack || error?.message || String(error);
  } finally {
    if (patchedSession) { try { await patchedSession.release(); } catch {} }
    if (baselineSession) { try { await baselineSession.release(); } catch {} }
    if (lock) releaseHeavyModelTest("full CLIP-L Personal Ami patch test");
    button.disabled = false;
  }
});
