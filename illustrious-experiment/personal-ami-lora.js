const MODEL_BASE = "https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/resolve/main/";
const COMPONENTS = ["text_encoder", "text_encoder_2", "unet_a", "unet_b", "unet_c"];
const EXPECTED = { text_encoder: 66, text_encoder_2: 192, unet_a: 350, unet_b: 306, unet_c: 66 };
const FINAL_CLIP_L = /^text_model_encoder_layers_11_(?:mlp_fc1|mlp_fc2|self_attn_(?:k_proj|out_proj|q_proj|v_proj))$/;

function groupForStem(stem) {
  if (stem.startsWith("lora_te1_")) return { component: "text_encoder", module: stem.slice(9) };
  if (stem.startsWith("lora_te2_")) return { component: "text_encoder_2", module: stem.slice(9) };
  if (stem.startsWith("lora_unet_")) return { component: "unet", module: stem.slice(10) };
  return null;
}

function normalizedIndexName(name) {
  const base = name.endsWith(".weight") ? name.slice(0, -7) : name;
  return base.replaceAll(".", "_");
}

function kohyaUnetToDiffusersName(module) {
  let key = module.replaceAll("_", ".");
  let match = key.match(/^input\.blocks\.(\d+)\.(\d+)(?:\.(.*))?$/);
  if (match) {
    const block = Number(match[1]), part = Number(match[2]), rest = match[3] || "";
    if (block === 0 && part === 0) key = "conv_in" + (rest ? "." + rest : "");
    else if (block > 0 && block % 3 === 0 && part === 0) {
      const stage = block / 3 - 1, suffix = rest.replace(/^op\.?/, "");
      key = "down_blocks." + stage + ".downsamplers.0.conv" + (suffix ? "." + suffix : "");
    } else if (block > 0) {
      const stage = Math.floor((block - 1) / 3), layer = (block - 1) % 3;
      if (layer > 1) return null;
      const collection = part === 0 ? "resnets" : part === 1 ? "attentions" : null;
      if (!collection) return null;
      key = "down_blocks." + stage + "." + collection + "." + layer + (rest ? "." + rest : "");
    } else return null;
  }
  match = key.match(/^output\.blocks\.(\d+)\.(\d+)(?:\.(.*))?$/);
  if (match) {
    const block = Number(match[1]), part = Number(match[2]), layer = block % 3;
    const stage = Math.floor(block / 3), rest = match[3] || "";
    let prefix;
    if (part === 0) prefix = "up_blocks." + stage + ".resnets." + layer;
    else if (part === 1) prefix = "up_blocks." + stage + ".attentions." + layer;
    else if (part === 2 && layer === 2) prefix = "up_blocks." + stage + ".upsamplers.0";
    else return null;
    key = prefix + (rest ? "." + rest : "");
  }
  match = key.match(/^middle\.block\.(\d+)(?:\.(.*))?$/);
  if (match) {
    const block = Number(match[1]), rest = match[2] || "";
    const target = block === 0 ? "mid_block.resnets.0" :
      block === 1 ? "mid_block.attentions.0" :
      block === 2 ? "mid_block.resnets.1" : null;
    if (!target) return null;
    key = target + (rest ? "." + rest : "");
  }
  key = key
    .replace(/^out\.2(?=\.|$)/, "conv_out")
    .replace(/^time\.embed\.0(?=\.|$)/, "time_embedding.linear_1")
    .replace(/^time\.embed\.2(?=\.|$)/, "time_embedding.linear_2")
    .replace(/^label\.emb\.0\.0(?=\.|$)/, "add_embedding.linear_1")
    .replace(/^label\.emb\.0\.2(?=\.|$)/, "add_embedding.linear_2")
    .replaceAll("input.blocks", "down_blocks")
    .replaceAll("down.blocks", "down_blocks")
    .replaceAll("middle.block", "mid_block")
    .replaceAll("mid.block", "mid_block")
    .replaceAll("output.blocks", "up_blocks")
    .replaceAll("up.blocks", "up_blocks")
    .replaceAll("transformer.blocks", "transformer_blocks")
    .replaceAll("proj.in", "proj_in")
    .replaceAll("proj.out", "proj_out")
    .replaceAll("to.q", "to_q")
    .replaceAll("to.k", "to_k")
    .replaceAll("to.v", "to_v")
    .replaceAll("to.out.0", "to_out.0")
    .replaceAll("emb.layers.1", "time_emb_proj")
    .replaceAll("time.emb.proj", "time_emb_proj")
    .replaceAll("in.layers.0", "norm1")
    .replaceAll("in.layers.2", "conv1")
    .replaceAll("out.layers.0", "norm2")
    .replaceAll("out.layers.3", "conv2")
    .replaceAll("skip.connection", "conv_shortcut")
    .replaceAll("conv.shortcut", "conv_shortcut")
    .replaceAll("downsamplers.0.op", "downsamplers.0.conv")
    .replaceAll("conv.in", "conv_in")
    .replaceAll("conv.out", "conv_out");
  return key;
}

function candidatesFor(group) {
  if (group.component !== "unet") return [group.module];
  const converted = kohyaUnetToDiffusersName(group.module);
  return [...new Set([converted && normalizedIndexName(converted), normalizedIndexName(group.module)].filter(Boolean))];
}

async function verifiedJson(path, fileMap) {
  const record = fileMap.get(path);
  if (!record) throw new Error("Manifest is missing " + path + ".");
  const response = await fetch(MODEL_BASE + path, { credentials: "omit", cache: "force-cache" });
  if (!response.ok) throw new Error(path + " returned HTTP " + response.status + ".");
  const buffer = await response.arrayBuffer();
  if (record.size !== undefined && buffer.byteLength !== record.size) throw new Error(path + " size mismatch.");
  if (record.sha256) {
    const digest = await crypto.subtle.digest("SHA-256", buffer);
    const actual = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, "0")).join("");
    if (actual !== record.sha256) throw new Error(path + " SHA-256 mismatch.");
  }
  return JSON.parse(new TextDecoder().decode(buffer));
}

function readHeader(file) {
  return (async () => {
    if (!file || file.size < 8) throw new Error("Select a valid Personal Ami .safetensors file.");
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
  })();
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
      !info.shape.every(value => Number.isSafeInteger(value) && value >= 0) ||
      !Array.isArray(info.data_offsets) || info.data_offsets.length !== 2 ||
      !info.data_offsets.every(value => Number.isSafeInteger(value) && value >= 0)) {
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
  if (typeof value === "number" && Number.isInteger(value) && value >= 0 && value < component.shards.length) {
    return component.shards[value];
  }
  if (typeof value === "string") {
    const normalized = value.replaceAll("\\", "/");
    const exact = component.shards.find(path => path === normalized);
    if (exact) return exact;
    const basename = normalized.split("/").pop();
    const matches = component.shards.filter(path => path === basename || path.endsWith("/" + basename));
    if (matches.length === 1) return matches[0];
  }
  if (component.shards.length === 1 && value == null) return component.shards[0];
  throw new Error("Cannot resolve index shard " + JSON.stringify(value) + ".");
}

function validatePair(pair, entry, stem) {
  const downShape = pair.down.shape, upShape = pair.up.shape, target = entry.torchShape;
  if (!Array.isArray(target) || (target.length !== 2 && target.length !== 4) ||
      downShape.length !== target.length || upShape.length !== target.length ||
      downShape[0] !== upShape[1]) throw new Error("Unsupported LoRA layout for " + stem + ".");
  const rank = downShape[0];
  if (target.length === 2) {
    if (downShape[1] !== target[1] || upShape[0] !== target[0]) throw new Error("Linear dimensions mismatch for " + stem + ".");
  } else {
    if (downShape[1] !== target[1] || upShape[0] !== target[0] ||
        downShape[2] !== target[2] || downShape[3] !== target[3] ||
        upShape[2] !== 1 || upShape[3] !== 1) throw new Error("Conv2D dimensions mismatch for " + stem + ".");
    if (entry.transposed) throw new Error("Transposed Conv2D storage is not supported for " + stem + ".");
  }
  return rank;
}

function patchLinear(buffer, down, up, shape, transposed, scale, label) {
  const [outFeatures, inFeatures] = shape;
  const rank = down.length / inFeatures;
  if (!Number.isInteger(rank) || up.length !== outFeatures * rank || buffer.byteLength !== outFeatures * inFeatures * 2) {
    throw new Error("Linear byte/shape mismatch for " + label + ".");
  }
  const base = new Float16Array(buffer);
  const patched = new Float16Array(base.length);
  const row = new Float32Array(inFeatures);
  for (let out = 0; out < outFeatures; out++) {
    row.fill(0);
    for (let r = 0; r < rank; r++) {
      const coefficient = up[out * rank + r], offset = r * inFeatures;
      for (let input = 0; input < inFeatures; input++) row[input] += coefficient * down[offset + input];
    }
    for (let input = 0; input < inFeatures; input++) {
      const index = transposed ? input * outFeatures + out : out * inFeatures + input;
      const value = Number(base[index]) + row[input] * scale;
      if (!Number.isFinite(value)) throw new Error("Non-finite patched weight in " + label + ".");
      patched[index] = value;
      if (!Number.isFinite(patched[index])) throw new Error("FP16 overflow while patching " + label + ".");
    }
  }
  return new Uint8Array(patched.buffer);
}

function patchConv2d(buffer, down, up, shape, scale, label) {
  const [outChannels, inChannels, height, width] = shape;
  const rank = down.length / (inChannels * height * width);
  if (!Number.isInteger(rank) || up.length !== outChannels * rank ||
      buffer.byteLength !== outChannels * inChannels * height * width * 2) {
    throw new Error("Conv2D byte/shape mismatch for " + label + ".");
  }
  const base = new Float16Array(buffer);
  const patched = new Float16Array(base.length);
  const kernelSize = height * width;
  const inputSize = inChannels * kernelSize;
  const row = new Float32Array(inputSize);
  for (let out = 0; out < outChannels; out++) {
    row.fill(0);
    for (let r = 0; r < rank; r++) {
      const coefficient = up[out * rank + r], offset = r * inputSize;
      for (let input = 0; input < inputSize; input++) row[input] += coefficient * down[offset + input];
    }
    const outputOffset = out * inputSize;
    for (let input = 0; input < inputSize; input++) {
      const value = Number(base[outputOffset + input]) + row[input] * scale;
      if (!Number.isFinite(value)) throw new Error("Non-finite patched Conv2D weight in " + label + ".");
      patched[outputOffset + input] = value;
      if (!Number.isFinite(patched[outputOffset + input])) throw new Error("FP16 overflow while patching " + label + ".");
    }
  }
  return new Uint8Array(patched.buffer);
}

async function patchShard(file, dataStart, component, shard, targets, fileMap, strength, onProgress, checkCancelled) {
  const record = fileMap.get(shard);
  if (!record || !Number.isSafeInteger(record.size)) throw new Error("Manifest is missing shard size for " + shard + ".");
  const response = await fetch(MODEL_BASE + shard, { credentials: "omit", cache: "force-cache" });
  if (!response.ok) throw new Error("Shard download failed: " + shard + " · HTTP " + response.status);
  const base = await response.blob();
  if (base.size !== record.size) throw new Error("Shard size mismatch for " + shard + ".");
  if (record.sha256) {
    const digest = await crypto.subtle.digest("SHA-256", await base.arrayBuffer());
    const actual = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, "0")).join("");
    if (actual !== record.sha256) throw new Error("SHA-256 mismatch for " + shard + ".");
  }
  targets.sort((a, b) => a.offset - b.offset);
  const parts = [];
  let cursor = 0;
  for (let i = 0; i < targets.length; i++) {
    checkCancelled();
    const target = targets[i];
    if (target.offset < cursor || target.offset + target.length > base.size) {
      throw new Error("Overlapping or out-of-range patch target " + target.target + ".");
    }
    parts.push(base.slice(cursor, target.offset));
    const pair = target.pair;
    const down = await readTensor(file, dataStart, pair.downName, pair.down);
    const up = await readTensor(file, dataStart, pair.upName, pair.up);
    const alpha = await readTensor(file, dataStart, pair.alphaName, pair.alpha);
    if (alpha.length !== 1) throw new Error("LoRA alpha must be scalar for " + target.stem + ".");
    const rank = validatePair(pair, target.entry, target.stem);
    const scale = alpha[0] / rank * strength;
    const targetBuffer = await base.slice(target.offset, target.offset + target.length).arrayBuffer();
    const patched = target.entry.torchShape.length === 2
      ? patchLinear(targetBuffer, down, up, target.entry.torchShape, !!target.entry.transposed, scale, target.target)
      : patchConv2d(targetBuffer, down, up, target.entry.torchShape, scale, target.target);
    parts.push(patched);
    cursor = target.offset + target.length;
    onProgress(target);
    if ((i + 1) % 16 === 0) await new Promise(resolve => requestAnimationFrame(() => resolve()));
  }
  parts.push(base.slice(cursor));
  return new Blob(parts, { type: "application/octet-stream" });
}

export async function preparePersonalAmiComponent(file, manifest, componentName, strength, log, checkCancelled) {
  if (!COMPONENTS.includes(componentName)) throw new Error("Unknown Personal Ami component " + componentName + ".");
  if (!Number.isFinite(strength) || strength < 0 || strength > 2) throw new Error("Personal Ami strength must be between 0 and 2.");
  const component = manifest.components?.[componentName];
  if (!component?.onnx || !component.weightsIndex || !Array.isArray(component.shards)) {
    throw new Error("Model manifest component is incomplete: " + componentName + ".");
  }
  const { header, dataStart } = await readHeader(file);
  const fileMap = new Map((manifest.files || []).map(item => [item.path, item]));
  const index = await verifiedJson(component.weightsIndex, fileMap);
  const componentIndexes = new Map([[componentName, index]]);
  if (componentName.startsWith("unet_")) {
    for (const name of ["unet_a", "unet_b", "unet_c"]) {
      if (name === componentName) continue;
      const other = manifest.components?.[name];
      if (!other?.weightsIndex) throw new Error("Missing weight index for " + name + ".");
      componentIndexes.set(name, await verifiedJson(other.weightsIndex, fileMap));
    }
  }
  const lookups = new Map();
  for (const [name, value] of componentIndexes) {
    const lookup = new Map();
    for (const [moduleName, entry] of Object.entries(value.entries || {})) {
      lookup.set(normalizedIndexName(moduleName), { name: moduleName, entry });
    }
    lookups.set(name, lookup);
  }

  const pairs = new Map();
  for (const [name, info] of Object.entries(header)) {
    let side = null, stem = name;
    for (const [suffix, value] of [[".lora_down.weight", "down"], [".lora_up.weight", "up"]]) {
      if (name.endsWith(suffix)) { side = value; stem = name.slice(0, -suffix.length); break; }
    }
    if (!side) continue;
    const group = groupForStem(stem);
    if (!group || (group.component !== "unet" && group.component !== componentName)) continue;
    if (!pairs.has(stem)) pairs.set(stem, { stem, group });
    const pair = pairs.get(stem);
    pair[side] = info;
    pair[side + "Name"] = name;
  }

  const targetsByShard = new Map();
  let mapped = 0, skipped = 0, payloadBytes = 0;
  for (const pair of pairs.values()) {
    checkCancelled();
    if (!pair.down || !pair.up) throw new Error("Incomplete LoRA pair " + pair.stem + ".");
    const alphaName = pair.stem + ".alpha";
    pair.alpha = header[alphaName];
    pair.alphaName = alphaName;
    if (!pair.alpha) throw new Error("Missing alpha tensor for " + pair.stem + ".");
    const keys = candidatesFor(pair.group);
    let found = null;
    let owner = null;
    const searchComponents = pair.group.component === "unet"
      ? ["unet_a", "unet_b", "unet_c"]
      : [componentName];
    for (const name of searchComponents) {
      for (const key of keys) {
        found = lookups.get(name)?.get(key) || null;
        if (found) { owner = name; break; }
      }
      if (found) break;
    }
    if (!found) {
      if (componentName === "text_encoder" && FINAL_CLIP_L.test(pair.group.module)) { skipped++; continue; }
      throw new Error("No ONNX index entry for " + pair.stem + " (tried " + keys.join(" | ") + ").");
    }
    if (owner !== componentName) continue;
    const entry = found.entry;
    const rank = validatePair(pair, entry, pair.stem);
    const shard = resolveShard(entry, component);
    const record = fileMap.get(shard);
    const offset = Number(entry.offset), length = Number(entry.length);
    const expectedLength = entry.torchShape.reduce((a, b) => a * b, 1) * 2;
    if (!record || !Number.isSafeInteger(offset) || offset < 0 ||
        !Number.isSafeInteger(length) || length !== expectedLength || offset + length > record.size) {
      throw new Error("Invalid indexed byte range for " + found.name + ".");
    }
    const list = targetsByShard.get(shard) || [];
    list.push({ stem: pair.stem, pair, entry, target: found.name, offset, length, rank });
    targetsByShard.set(shard, list);
    mapped++;
    payloadBytes += (pair.down.data_offsets[1] - pair.down.data_offsets[0]) +
      (pair.up.data_offsets[1] - pair.up.data_offsets[0]) +
      (pair.alpha.data_offsets[1] - pair.alpha.data_offsets[0]);
  }

  const expected = EXPECTED[componentName];
  if (mapped !== expected || (componentName === "text_encoder" && skipped !== 6)) {
    throw new Error(componentName + " mapping count mismatch: expected " + expected +
      " mapped pairs" + (componentName === "text_encoder" ? " + 6 unindexed final-layer pairs" : "") +
      ", found " + mapped + " mapped and " + skipped + " skipped.");
  }

  log(componentName + ": mapped " + mapped + " pairs" +
    (skipped ? " · excluded " + skipped + " unindexed CLIP-L layer-11 pairs" : "") +
    " · reading " + (payloadBytes / 1024 / 1024).toFixed(2) + " MiB of local BF16 factors.");
  const patched = [];
  for (const shard of component.shards) {
    checkCancelled();
    const targets = targetsByShard.get(shard) || [];
    if (!targets.length) {
      patched.push({ path: shard.split("/").pop(), data: MODEL_BASE + shard });
      continue;
    }
    log(componentName + ": downloading/verifying " + shard + " and applying " + targets.length + " patches…");
    const blob = await patchShard(file, dataStart, component, shard, targets, fileMap, strength,
      () => {}, checkCancelled);
    patched.push({ path: shard.split("/").pop(), data: blob });
    log(componentName + ": patched " + targets.length + " initializers in " + shard + ".");
  }
  return { externalData: patched, mapped, skipped, payloadBytes };
}
