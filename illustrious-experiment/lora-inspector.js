const MAX_HEADER_BYTES = 64 * 1024 * 1024;
const DTYPE_BYTES = Object.freeze({
  BOOL: 1,
  U8: 1,
  I8: 1,
  F8_E5M2: 1,
  F8_E4M3: 1,
  F8_E4M3FN: 1,
  F8_E4M3FNUZ: 1,
  F8_E5M2FNUZ: 1,
  U16: 2,
  I16: 2,
  F16: 2,
  BF16: 2,
  U32: 4,
  I32: 4,
  F32: 4,
  U64: 8,
  I64: 8,
  F64: 8
});

const fileInput = document.querySelector("#loraFile");
const inspectButton = document.querySelector("#inspectLora");
const result = document.querySelector("#loraResult");
const summary = document.querySelector("#loraSummary");
const metadata = document.querySelector("#loraMetadata");
const tensorList = document.querySelector("#tensorList");
const fileName = document.querySelector("#fileName");
const status = document.querySelector("#inspectStatus");

const ADAPTER_SUFFIXES = [
  [".lora_down.weight", "down"],
  [".lora_up.weight", "up"],
  [".lora_A.weight", "down"],
  [".lora_B.weight", "up"]
];

function setStatus(message, type = "info") {
  status.textContent = message;
  status.dataset.type = type;
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  if (bytes < 1024 * 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + " MB";
  return (bytes / 1024 / 1024 / 1024).toFixed(2) + " GB";
}

function addRow(parent, label, value) {
  const row = document.createElement("div");
  row.className = "data-row";
  const labelNode = document.createElement("span");
  labelNode.className = "data-label";
  labelNode.textContent = label;
  const valueNode = document.createElement("span");
  valueNode.className = "data-value";
  valueNode.textContent = value;
  row.append(labelNode, valueNode);
  parent.append(row);
}

function tensorGroup(name) {
  const key = name.toLowerCase();
  if (key.includes("text_encoder_2") || key.includes("lora_te2_") || key.includes("lora_te2.")) return "Text encoder 2";
  if (key.includes("text_encoder") || key.includes("lora_te_") || key.includes("lora_te1_") || key.includes("lora_te1.")) return "Text encoder 1";
  if (key.includes("unet") || key.includes("lora_unet_") || key.includes("lora_unet.")) return "UNet";
  if (key.includes("transformer") || key.includes("lora_transformer_")) return "Transformer";
  if (key.includes("vae")) return "VAE";
  return "Other";
}

function pairInfo(name) {
  for (const [suffix, side] of ADAPTER_SUFFIXES) {
    if (name.endsWith(suffix)) {
      return { stem: name.slice(0, -suffix.length), side };
    }
  }
  return null;
}

function readMetadata(meta) {
  const preferred = [
    "ss_sd_model_name",
    "ss_base_model_version",
    "ss_network_module",
    "ss_network_dim",
    "ss_network_alpha",
    "ss_v2",
    "modelspec.architecture",
    "modelspec.title",
    "modelspec.description"
  ];
  const output = [];
  for (const key of preferred) {
    if (meta[key] !== undefined && meta[key] !== "") {
      let value = String(meta[key]);
      if (value.length > 300) value = value.slice(0, 297) + "...";
      output.push([key, value]);
    }
  }
  for (const [key, value] of Object.entries(meta)) {
    if (output.length >= 12) break;
    if (preferred.includes(key) || value === undefined || value === "") continue;
    if (/trigger|trained.?word|prompt|base.?model|architecture/i.test(key)) {
      let text = typeof value === "string" ? value : JSON.stringify(value);
      if (text.length > 300) text = text.slice(0, 297) + "...";
      output.push([key, text]);
    }
  }
  return output;
}

function validateTensorRecord(name, info, dataBytes) {
  if (!info || typeof info !== "object" || !Array.isArray(info.shape) || !Array.isArray(info.data_offsets)) {
    throw new Error("Invalid tensor record: " + name);
  }
  if (typeof info.dtype !== "string" || !Object.prototype.hasOwnProperty.call(DTYPE_BYTES, info.dtype)) {
    throw new Error("Tensor has an unsupported or missing dtype: " + name);
  }
  if (!info.shape.every(dim => Number.isSafeInteger(dim) && dim >= 0)) {
    throw new Error("Tensor has an invalid shape: " + name);
  }
  if (
    info.data_offsets.length !== 2 ||
    !info.data_offsets.every(value => Number.isSafeInteger(value) && value >= 0)
  ) {
    throw new Error("Tensor has invalid data offsets: " + name);
  }
  const [start, end] = info.data_offsets;
  if (end < start || end > dataBytes) {
    throw new Error("Tensor data extends beyond the file: " + name);
  }
  const elements = info.shape.reduce((product, dim) => product * BigInt(dim), 1n);
  const expectedBytes = elements * BigInt(DTYPE_BYTES[info.dtype]);
  if (BigInt(end - start) !== expectedBytes) {
    throw new Error("Tensor byte range does not match its shape and dtype: " + name);
  }
  return elements;
}

async function inspectSafetensors(file) {
  if (!file || !Number.isSafeInteger(file.size) || file.size < 8) {
    throw new Error("This file is too small to be a safetensors file.");
  }

  const prefix = await file.slice(0, 8).arrayBuffer();
  const headerLength = new DataView(prefix).getBigUint64(0, true);
  if (headerLength === 0n || headerLength > BigInt(MAX_HEADER_BYTES)) {
    throw new Error("The safetensors header length is invalid or exceeds the 64 MB safety limit.");
  }
  if (headerLength > BigInt(file.size - 8)) {
    throw new Error("The file ends before its declared safetensors header.");
  }

  const dataStart = 8 + Number(headerLength);
  const headerBytes = await file.slice(8, dataStart).arrayBuffer();
  let header;
  try {
    header = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(headerBytes));
  } catch {
    throw new Error("The safetensors header is not valid UTF-8 JSON.");
  }
  if (!header || typeof header !== "object" || Array.isArray(header)) {
    throw new Error("The safetensors header must be a JSON object.");
  }

  const meta = header.__metadata__ && typeof header.__metadata__ === "object" && !Array.isArray(header.__metadata__)
    ? header.__metadata__
    : {};
  const records = Object.entries(header).filter(([key]) => key !== "__metadata__");
  if (!records.length) throw new Error("No tensor records were found in the safetensors header.");

  const groups = new Map();
  const pairs = new Map();
  const flags = { dora: false, lokr: false, loha: false, oft: false };
  const examples = [];
  let totalParameters = 0n;

  for (const [name, info] of records) {
    const parameterCount = validateTensorRecord(name, info, file.size - dataStart);
    totalParameters += parameterCount;

    const group = tensorGroup(name);
    groups.set(group, (groups.get(group) || 0) + 1);

    const lower = name.toLowerCase();
    if (lower.includes("dora") || lower.includes("magnitude_vector")) flags.dora = true;
    if (lower.includes("lokr_") || lower.includes(".lokr")) flags.lokr = true;
    if (lower.includes("hada_") || lower.includes(".hada")) flags.loha = true;
    if (lower.includes("oft_") || lower.includes(".oft")) flags.oft = true;

    const pair = pairInfo(name);
    if (pair) {
      if (!pairs.has(pair.stem)) pairs.set(pair.stem, { down: null, up: null });
      pairs.get(pair.stem)[pair.side] = info.shape;
    }
    if (examples.length < 30) {
      examples.push({ name, shape: info.shape, dtype: info.dtype });
    }
  }

  let paired = 0;
  let incomplete = 0;
  let incompatiblePairs = 0;
  for (const pair of pairs.values()) {
    if (!pair.down || !pair.up) {
      incomplete++;
      continue;
    }
    paired++;
    // Standard Linear and Conv2d LoRAs both encode rank on down[0]/up[1].
    if (pair.down.length < 2 || pair.up.length < 2 || pair.down[0] !== pair.up[1]) {
      incompatiblePairs++;
    }
  }

  const metadataRows = readMetadata(meta);
  const extendedTypes = Object.entries(flags).filter(([, present]) => present).map(([type]) => type);
  let format = paired > 0
    ? "Standard low-rank A/B or down/up tensors detected"
    : "No standard LoRA tensor pairs detected";
  if (extendedTypes.length) format += "; extended adapter tensors: " + extendedTypes.join(", ");
  if (incompatiblePairs) format += "; " + incompatiblePairs + " pair(s) have mismatched rank dimensions";

  const familyHint = groups.has("Text encoder 2")
    ? "SDXL-like dual text encoders (strong hint)"
    : groups.has("UNet")
      ? "UNet adapter detected; SDXL vs. another UNet family still needs tensor-shape matching"
      : "Architecture not identifiable from these tensor names alone";

  return {
    fileName: file.name,
    fileSize: file.size,
    tensorCount: records.length,
    totalParameters: totalParameters.toLocaleString(),
    groups,
    pairs: paired,
    incomplete,
    incompatiblePairs,
    flags,
    format,
    familyHint,
    metadataRows,
    examples
  };
}

function renderReport(report) {
  summary.replaceChildren();
  metadata.replaceChildren();
  tensorList.replaceChildren();

  addRow(summary, "File size", formatBytes(report.fileSize));
  addRow(summary, "Tensors", String(report.tensorCount));
  addRow(summary, "Tensor parameters", report.totalParameters);
  addRow(summary, "Complete LoRA pairs", String(report.pairs));
  addRow(summary, "Incomplete pairs", String(report.incomplete));
  addRow(summary, "Rank-shape mismatches", String(report.incompatiblePairs));
  addRow(summary, "Format", report.format);
  addRow(summary, "Architecture hint", report.familyHint);

  for (const [group, count] of report.groups) addRow(summary, group, String(count));
  if (report.flags.dora) addRow(summary, "Adapter type", "DoRA tensors present");
  if (report.flags.lokr) addRow(summary, "Adapter type", "LoKr tensors present");
  if (report.flags.loha) addRow(summary, "Adapter type", "LoHa tensors present");
  if (report.flags.oft) addRow(summary, "Adapter type", "OFT tensors present");

  if (!report.metadataRows.length) addRow(metadata, "Metadata", "No useful training metadata in header");
  for (const [key, value] of report.metadataRows) addRow(metadata, key, value);

  for (const tensor of report.examples) {
    const item = document.createElement("li");
    const name = document.createElement("code");
    name.textContent = tensor.name;
    const shape = document.createElement("span");
    shape.className = "shape";
    shape.textContent = "[" + tensor.shape.join(", ") + "] · " + tensor.dtype;
    item.append(name, shape);
    tensorList.append(item);
  }
  result.hidden = false;
}

inspectButton.addEventListener("click", async () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    setStatus("Choose a .safetensors file first.", "error");
    return;
  }
  inspectButton.disabled = true;
  result.hidden = true;
  setStatus("Reading the safetensors header only. Tensor payloads are not uploaded or loaded.", "info");
  fileName.textContent = file.name;
  try {
    if (!file.name.toLowerCase().endsWith(".safetensors")) {
      throw new Error("Choose a .safetensors file. This inspector does not parse .ckpt or .pt files.");
    }
    const report = await inspectSafetensors(file);
    renderReport(report);
    setStatus("Header inspection complete. The file stayed in this browser tab.", "success");
  } catch (error) {
    setStatus(error.message || String(error), "error");
  } finally {
    inspectButton.disabled = false;
  }
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  fileName.textContent = file ? file.name + " (" + formatBytes(file.size) + ")" : "No file selected";
  result.hidden = true;
  if (file) setStatus("Ready to inspect local safetensors header.", "info");
});

const dropZone = document.querySelector("#dropZone");
dropZone.addEventListener("dragover", event => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", event => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  const file = event.dataTransfer && event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  fileInput.dispatchEvent(new Event("change"));
});

document.querySelector("#copyTrigger").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText("personal ami");
    setStatus("Copied trigger phrase: personal ami", "success");
  } catch {
    setStatus("Clipboard access was blocked. Trigger phrase: personal ami", "error");
  }
});
