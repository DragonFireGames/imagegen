const MODEL_BASE = "https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/resolve/main/";
const MANIFEST_URL = MODEL_BASE + "manifest.json";
const HEADER_LIMIT = 64 * 1024 * 1024;

const fileInput = document.querySelector("#loraFile");
const button = document.querySelector("#mapLoraWeights");
const status = document.querySelector("#loraMapStatus");
const summary = document.querySelector("#loraMapSummary");
const output = document.querySelector("#loraMapOutput");

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

const SUFFIXES = [
  [".lora_down.weight", "down"],
  [".lora_up.weight", "up"],
  [".lora_A.weight", "down"],
  [".lora_B.weight", "up"]
];

function splitPairKey(name) {
  for (const [suffix, side] of SUFFIXES) {
    if (name.endsWith(suffix)) return { stem: name.slice(0, -suffix.length), side };
  }
  return null;
}

function readHeader(file) {
  return (async () => {
    if (!file || file.size < 8) throw new Error("Choose a valid .safetensors file first.");
    const prefix = await file.slice(0, 8).arrayBuffer();
    const headerLength = new DataView(prefix).getBigUint64(0, true);
    if (headerLength === 0n || headerLength > BigInt(HEADER_LIMIT)) {
      throw new Error("Safetensors header is invalid or exceeds the 64 MiB safety limit.");
    }
    if (headerLength > BigInt(file.size - 8)) throw new Error("File ends before its declared safetensors header.");
    const dataStart = 8 + Number(headerLength);
    const bytes = await file.slice(8, dataStart).arrayBuffer();
    let header;
    try {
      header = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
    } catch {
      throw new Error("Safetensors header is not valid UTF-8 JSON.");
    }
    if (!header || typeof header !== "object" || Array.isArray(header)) {
      throw new Error("Safetensors header must be a JSON object.");
    }

    const tensors = new Map();
    const pairs = new Map();
    for (const [name, info] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      if (!info || !Array.isArray(info.shape) || !info.shape.every(x => Number.isSafeInteger(x) && x >= 0)) {
        throw new Error("Invalid tensor shape in safetensors header: " + name);
      }
      if (!Array.isArray(info.data_offsets) || info.data_offsets.length !== 2 ||
          !info.data_offsets.every(x => Number.isSafeInteger(x) && x >= 0) ||
          info.data_offsets[1] < info.data_offsets[0] ||
          info.data_offsets[1] > file.size - dataStart) {
        throw new Error("Invalid tensor data offsets in safetensors header: " + name);
      }
      tensors.set(name, info.shape);
      const part = splitPairKey(name);
      if (part) {
        if (!pairs.has(part.stem)) pairs.set(part.stem, { down: null, up: null });
        pairs.get(part.stem)[part.side] = info.shape;
      }
    }
    return { tensors, pairs, metadata: header.__metadata__ || {}, headerBytes: dataStart };
  })();
}

function groupForStem(stem) {
  if (stem.startsWith("lora_te1_")) return { component: "text_encoder", label: "CLIP-L", module: stem.slice("lora_te1_".length) };
  if (stem.startsWith("lora_te2_")) return { component: "text_encoder_2", label: "CLIP-bigG", module: stem.slice("lora_te2_".length) };
  if (stem.startsWith("lora_unet_")) return { component: "unet", label: "UNet", module: stem.slice("lora_unet_".length) };
  return null;
}

function normalizedIndexName(name) {
  const base = name.endsWith(".weight") ? name.slice(0, -".weight".length) : name;
  return base.replaceAll(".", "_");
}

function kohyaUnetToDiffusersName(module) {
  let key = module.replaceAll("_", ".");

  // A Kohya/SGM SDXL UNet uses Kohya/SGM block indices. Convert those indices
  // to the actual SDXL Diffusers module paths used by weights-index.json.
  let match = key.match(/^input\.blocks\.(\d+)\.(\d+)(?:\.(.*))?$/);
  if (match) {
    const block = Number(match[1]);
    const part = Number(match[2]);
    const rest = match[3] || "";
    if (block === 0 && part === 0) {
      key = "conv_in" + (rest ? "." + rest : "");
    } else if (block > 0 && block % 3 === 0 && part === 0) {
      const stage = block / 3 - 1;
      const suffix = rest.replace(/^op\.?/, "");
      key = "down_blocks." + stage + ".downsamplers.0.conv" +
        (suffix ? "." + suffix : "");
    } else if (block > 0) {
      const stage = Math.floor((block - 1) / 3);
      const layer = (block - 1) % 3;
      if (layer > 1) return null;
      const collection = part === 0 ? "resnets" : part === 1 ? "attentions" : null;
      if (!collection) return null;
      key = "down_blocks." + stage + "." + collection + "." + layer +
        (rest ? "." + rest : "");
    } else {
      return null;
    }
  }

  match = key.match(/^output\.blocks\.(\d+)\.(\d+)(?:\.(.*))?$/);
  if (match) {
    const block = Number(match[1]);
    const part = Number(match[2]);
    const layer = block % 3;
    const stage = Math.floor(block / 3);
    const rest = match[3] || "";
    let prefix;
    if (part === 0) prefix = "up_blocks." + stage + ".resnets." + layer;
    else if (part === 1) prefix = "up_blocks." + stage + ".attentions." + layer;
    else if (part === 2 && layer === 2) prefix = "up_blocks." + stage + ".upsamplers.0";
    else return null;
    key = prefix + (rest ? "." + rest : "");
  }

  match = key.match(/^middle\.block\.(\d+)(?:\.(.*))?$/);
  if (match) {
    const block = Number(match[1]);
    const rest = match[2] || "";
    const target = block === 0 ? "mid_block.resnets.0" :
      block === 1 ? "mid_block.attentions.0" :
      block === 2 ? "mid_block.resnets.1" : null;
    if (!target) return null;
    key = target + (rest ? "." + rest : "");
  }

  // Standalone SDXL UNet modules.
  key = key
    .replace(/^out\.2(?=\.|$)/, "conv_out")
    .replace(/^time\.embed\.0(?=\.|$)/, "time_embedding.linear_1")
    .replace(/^time\.embed\.2(?=\.|$)/, "time_embedding.linear_2")
    .replace(/^label\.emb\.0\.0(?=\.|$)/, "add_embedding.linear_1")
    .replace(/^label\.emb\.0\.2(?=\.|$)/, "add_embedding.linear_2");

  // Convert SGM/CompVis submodule spellings to Diffusers names.
  key = key
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

function unetIndexCandidates(module) {
  const candidates = [normalizedIndexName(module)];
  const converted = kohyaUnetToDiffusersName(module);
  if (converted) candidates.unshift(normalizedIndexName(converted));
  return [...new Set(candidates)];
}

function compareShapes(down, up, entry) {
  const torch = entry.torchShape;
  if (!Array.isArray(torch) || !torch.length) return { ok: false, reason: "no torchShape in index" };
  if (down.length < 2 || up.length < 2 || down[0] !== up[1]) {
    return { ok: false, reason: "down/up rank dimension mismatch" };
  }
  const rank = down[0];
  if (torch.length === 2 && down.length === 2 && up.length === 2) {
    if (down[1] !== torch[1]) return { ok: false, reason: "down input dimension mismatch" };
    if (up[0] !== torch[0]) return { ok: false, reason: "up output dimension mismatch" };
    return { ok: true, rank, layout: "linear", transposed: !!entry.transposed };
  }
  if (torch.length === 4 && down.length === 4 && up.length === 4) {
    if (down[1] !== torch[1] || up[0] !== torch[0]) return { ok: false, reason: "conv in/out dimension mismatch" };
    if (down[2] !== torch[2] || down[3] !== torch[3]) return { ok: false, reason: "conv down kernel mismatch" };
    if (up[2] !== 1 || up[3] !== 1) return { ok: false, reason: "conv up kernel is not 1×1" };
    return { ok: true, rank, layout: "conv2d", transposed: !!entry.transposed };
  }
  return { ok: false, reason: "unsupported LoRA/target tensor layout" };
}

async function fetchVerifiedIndex(path, fileMap) {
  const expected = fileMap.get(path);
  if (!expected) throw new Error("Manifest has no file record for " + path + ".");
  const response = await fetch(MODEL_BASE + path, { credentials: "omit" });
  if (!response.ok) throw new Error("Index request failed for " + path + ": HTTP " + response.status);
  const buffer = await response.arrayBuffer();
  if (expected.size !== undefined && buffer.byteLength !== expected.size) {
    throw new Error(path + " size mismatch: expected " + expected.size + ", received " + buffer.byteLength + ".");
  }
  if (expected.sha256) {
    const hash = await crypto.subtle.digest("SHA-256", buffer);
    const actual = Array.from(new Uint8Array(hash), value => value.toString(16).padStart(2, "0")).join("");
    if (actual !== expected.sha256) throw new Error(path + " SHA-256 mismatch.");
  }
  return JSON.parse(new TextDecoder().decode(buffer));
}

button.addEventListener("click", async () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    setStatus("Choose the LoRA .safetensors file in the inspector above first.", "error");
    return;
  }

  button.disabled = true;
  summary.replaceChildren();
  summary.hidden = false;
  output.textContent = "";
  output.hidden = false;

  try {
    setStatus("Reading only the local safetensors header. Adapter tensor payloads remain untouched…");
    const started = performance.now();
    const adapter = await readHeader(file);

    setStatus("Fetching the model manifest and five small weight indexes. No ONNX graphs or weight shards are downloaded…");
    const manifestResponse = await fetch(MANIFEST_URL, { credentials: "omit" });
    if (!manifestResponse.ok) throw new Error("Model manifest request failed: HTTP " + manifestResponse.status);
    const manifest = await manifestResponse.json();
    const fileMap = new Map((manifest.files || []).map(item => [item.path, item]));

    const componentNames = ["text_encoder", "text_encoder_2", "unet_a", "unet_b", "unet_c"];
    const indexes = new Map();
    let indexBytes = 0;
    for (const name of componentNames) {
      const component = manifest.components?.[name];
      if (!component?.weightsIndex) {
        indexes.set(name, null);
        continue;
      }
      const record = fileMap.get(component.weightsIndex);
      if (!record) throw new Error("Manifest file record is missing for " + component.weightsIndex + ".");
      indexBytes += record.size || 0;
      indexes.set(name, await fetchVerifiedIndex(component.weightsIndex, fileMap));
    }

    const lookup = new Map();
    for (const name of componentNames) {
      const index = indexes.get(name);
      if (!index?.entries) continue;
      for (const [moduleName, entry] of Object.entries(index.entries)) {
        const key = normalizedIndexName(moduleName);
        if (!lookup.has(name)) lookup.set(name, new Map());
        lookup.get(name).set(key, { moduleName, entry });
      }
    }

    const counts = {
      total: 0,
      matched: 0,
      shapeValid: 0,
      missingName: 0,
      unindexedFinalClipL: 0,
      missingPair: 0,
      shapeMismatch: 0,
      unsupportedGroup: 0,
      transposed: 0,
      linear: 0,
      conv2d: 0
    };
    const issues = [];
    const mappedByComponent = new Map();

    for (const [stem, pair] of adapter.pairs) {
      counts.total++;
      if (!pair.down || !pair.up) {
        counts.missingPair++;
        if (issues.length < 80) issues.push("INCOMPLETE " + stem);
        continue;
      }
      const group = groupForStem(stem);
      if (!group) {
        counts.unsupportedGroup++;
        if (issues.length < 80) issues.push("UNSUPPORTED PREFIX " + stem);
        continue;
      }
      const componentNamesToTry = group.component === "unet"
        ? ["unet_a", "unet_b", "unet_c"]
        : [group.component];
      const keysToTry = group.component === "unet"
        ? unetIndexCandidates(group.module)
        : [group.module];
      let found = null;
      let foundComponent = null;
      for (const componentName of componentNamesToTry) {
        for (const key of keysToTry) {
          const entry = lookup.get(componentName)?.get(key);
          if (entry) {
            found = entry;
            foundComponent = componentName;
            break;
          }
        }
        if (found) break;
      }
      if (!found) {
        const isUnindexedFinalClipLBlock =
          group.component === "text_encoder" &&
          /^text_model_encoder_layers_11_(?:mlp_fc1|mlp_fc2|self_attn_(?:k_proj|out_proj|q_proj|v_proj))$/.test(group.module);
        if (isUnindexedFinalClipLBlock) {
          counts.unindexedFinalClipL++;
          if (issues.length < 80) {
            issues.push("UNINDEXED FINAL CLIP-L BLOCK " + stem +
              " · not present in the graph index; consistent with the graph exposing penultimate_hidden_state");
          }
        } else {
          counts.missingName++;
          if (issues.length < 80) {
            issues.push("NO INDEX MATCH " + stem + " → tried: " + keysToTry.join(" | "));
          }
        }
        continue;
      }
      counts.matched++;
      const shape = compareShapes(pair.down, pair.up, found.entry);
      if (!shape.ok) {
        counts.shapeMismatch++;
        if (issues.length < 80) issues.push("SHAPE MISMATCH " + stem + " → " + found.moduleName +
          " · down [" + pair.down.join(",") + "] up [" + pair.up.join(",") +
          "] target [" + (found.entry.torchShape || []).join(",") + "] · " + shape.reason);
        continue;
      }
      counts.shapeValid++;
      if (shape.transposed) counts.transposed++;
      counts[shape.layout]++;
      if (!mappedByComponent.has(foundComponent)) mappedByComponent.set(foundComponent, []);
      mappedByComponent.get(foundComponent).push({
        stem,
        moduleName: found.moduleName,
        initializer: found.entry.initializer,
        shard: found.entry.shard,
        offset: found.entry.offset,
        length: found.entry.length,
        transposed: shape.transposed,
        rank: shape.rank
      });
    }

    const lines = [
      "Adapter: " + file.name,
      "Model pack: " + (manifest.packageId || "WAI Illustrious v17"),
      "UNet naming: Kohya/SGM SDXL block paths converted to Diffusers index names",
      "Header read: " + formatBytes(adapter.headerBytes) + " · tensor payload bytes read: 0",
      "Weight-index JSON: " + formatBytes(indexBytes) + " · all listed indexes size/SHA-256 verified where hashes are provided",
      "",
      "Pair mapping summary:",
      "  Pairs found in header: " + counts.total,
      "  Matching index names: " + counts.matched,
      "  Matching names + shape-compatible: " + counts.shapeValid,
      "  Expected unindexed CLIP-L layer-11 pairs: " + counts.unindexedFinalClipL,
      "  Unexpected missing index name: " + counts.missingName,
      "  Incomplete down/up pair: " + counts.missingPair,
      "  Shape mismatches: " + counts.shapeMismatch,
      "  Unsupported prefixes: " + counts.unsupportedGroup,
      "  Linear targets: " + counts.linear,
      "  Conv2D targets: " + counts.conv2d,
      "  Mapped matrices requiring transpose on ONNX storage: " + counts.transposed,
      "",
      "Mapped by component:"
    ];

    for (const [name, entries] of mappedByComponent) {
      lines.push("  " + name + ": " + entries.length + " shape-compatible LoRA pairs");
    }
    lines.push("", "Sample mapped targets:");
    let samples = 0;
    for (const [name, entries] of mappedByComponent) {
      for (const item of entries) {
        if (samples++ >= 24) break;
        lines.push("  [" + name + "] " + item.stem + " → " + item.moduleName +
          " → " + item.initializer + " · rank " + item.rank +
          (item.transposed ? " · transpose delta for ONNX" : ""));
      }
      if (samples >= 24) break;
    }
    lines.push("", "First mapping issues:");
    lines.push(...(issues.length ? issues : ["  None."]));
    lines.push("", "Important: this is a header/name/shape mapping test only. It does not read LoRA tensor values, calculate A×B deltas, apply transposes, patch ONNX shards, or prove output equivalence to Diffusers.");

    output.textContent = lines.join("\n");
    addRow("LoRA file", file.name + " · " + formatBytes(file.size));
    addRow("Pairs found", String(counts.total));
    addRow("Matching index names", String(counts.matched));
    addRow("Shape-compatible pairs", String(counts.shapeValid));
    addRow("Unindexed final CLIP-L layer-11 pairs", String(counts.unindexedFinalClipL));
    addRow("Unexpected missing names / shape mismatches", counts.missingName + " / " + counts.shapeMismatch);
    addRow("Transpose-required targets", String(counts.transposed));
    addRow("Index JSON downloaded", formatBytes(indexBytes));
    addRow("Adapter tensor payload downloaded/read", "0 bytes");
    addRow("Inspection time", ((performance.now() - started) / 1000).toFixed(1) + " s");

    setStatus("LoRA header-to-index mapping completed. Review the detailed per-pair report; shape-compatible mappings are necessary but not sufficient for verified LoRA application.", "success");
  } catch (error) {
    console.error("[Illustrious] LoRA mapping inspection failed:", error);
    setStatus("LoRA mapping inspection failed: " + (error?.message || String(error)), "error");
  } finally {
    button.disabled = false;
  }
});
