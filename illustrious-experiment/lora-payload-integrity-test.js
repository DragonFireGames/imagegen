const HEADER_LIMIT = 64 * 1024 * 1024;
const CHUNK_BYTES = 4 * 1024 * 1024;

const fileInput = document.querySelector("#loraFile");
const button = document.querySelector("#verifyAllLoraPayloads");
const status = document.querySelector("#loraIntegrityStatus");
const summary = document.querySelector("#loraIntegritySummary");

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

async function readLocalHeader(file) {
  if (!file || file.size < 8) throw new Error("Choose a valid .safetensors file first.");
  const prefix = await file.slice(0, 8).arrayBuffer();
  const headerLength = new DataView(prefix).getBigUint64(0, true);
  if (headerLength === 0n || headerLength > BigInt(HEADER_LIMIT)) {
    throw new Error("Safetensors header is invalid or exceeds 64 MiB.");
  }
  if (headerLength > BigInt(file.size - 8)) throw new Error("File ends before its declared header.");
  const dataStart = 8 + Number(headerLength);
  const bytes = await file.slice(8, dataStart).arrayBuffer();
  const header = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  if (!header || typeof header !== "object" || Array.isArray(header)) {
    throw new Error("Safetensors header must be a JSON object.");
  }
  return { header, dataStart };
}

button.addEventListener("click", async () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    setStatus("Choose the LoRA .safetensors file above first.", "error");
    return;
  }

  button.disabled = true;
  summary.replaceChildren();
  summary.hidden = false;

  try {
    const started = performance.now();
    setStatus("Reading the local safetensors header and validating all BF16 tensor ranges…");
    const { header, dataStart } = await readLocalHeader(file);
    const dataBytes = file.size - dataStart;
    const tensors = [];
    let totalParameters = 0n;
    let expectedPayloadBytes = 0n;

    for (const [name, info] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      if (!info || info.dtype !== "BF16") {
        throw new Error("Expected BF16 tensor dtype; found " + (info?.dtype || "missing") + " for " + name + ".");
      }
      if (!Array.isArray(info.shape) ||
          !info.shape.every(dimension => Number.isSafeInteger(dimension) && dimension >= 0)) {
        throw new Error("Invalid tensor shape in header: " + name + ".");
      }
      if (!Array.isArray(info.data_offsets) || info.data_offsets.length !== 2 ||
          !info.data_offsets.every(offset => Number.isSafeInteger(offset) && offset >= 0)) {
        throw new Error("Invalid data offsets in header: " + name + ".");
      }

      const [start, end] = info.data_offsets;
      const elements = info.shape.reduce((product, dimension) => product * BigInt(dimension), 1n);
      const expectedBytes = elements * 2n;
      if (end < start || end > dataBytes || BigInt(end - start) !== expectedBytes) {
        throw new Error("Tensor byte range does not match shape × BF16 size: " + name + ".");
      }

      tensors.push({ name, start, end });
      totalParameters += elements;
      expectedPayloadBytes += expectedBytes;
    }

    if (!tensors.length) throw new Error("No tensor records found in the safetensors header.");
    if (expectedPayloadBytes !== BigInt(dataBytes)) {
      throw new Error("Header tensor ranges account for " + expectedPayloadBytes +
        " bytes, but the file data block has " + dataBytes + " bytes.");
    }

    tensors.sort((a, b) => a.start - b.start);
    let previousEnd = 0;
    for (const tensor of tensors) {
      if (tensor.start !== previousEnd) {
        throw new Error("Safetensors payload ranges overlap or leave a gap before " + tensor.name + ".");
      }
      previousEnd = tensor.end;
    }
    if (previousEnd !== dataBytes) throw new Error("Safetensors tensor ranges do not cover the entire data block.");

    setStatus("Scanning " + (dataBytes / 1024 / 1024).toFixed(1) +
      " MiB of local BF16 tensor bytes in 4 MiB chunks. Nothing is uploaded…");

    let scannedBytes = 0;
    let scannedValues = 0;
    let chunks = 0;
    while (scannedBytes < dataBytes) {
      const end = Math.min(dataBytes, scannedBytes + CHUNK_BYTES);
      const buffer = await file.slice(dataStart + scannedBytes, dataStart + end).arrayBuffer();
      if (buffer.byteLength % 2 !== 0) throw new Error("BF16 chunk has an odd byte length.");
      const view = new DataView(buffer);
      for (let offset = 0; offset < buffer.byteLength; offset += 2) {
        const bits = view.getUint16(offset, true);
        // BF16 uses an 8-bit exponent. Exponent 255 encodes infinity or NaN.
        if ((bits & 0x7f80) === 0x7f80) {
          throw new Error("Found an Inf/NaN BF16 payload value at data byte offset " +
            (scannedBytes + offset) + ".");
        }
        scannedValues++;
      }
      scannedBytes = end;
      chunks++;
      setStatus("Scanned " + (scannedBytes / 1024 / 1024).toFixed(1) + " / " +
        (dataBytes / 1024 / 1024).toFixed(1) + " MiB of local tensor payload…");
      await new Promise(resolve => requestAnimationFrame(resolve));
    }

    addRow("Local file", file.name);
    addRow("Tensor records", tensors.length.toLocaleString());
    addRow("Tensor parameters", totalParameters.toLocaleString());
    addRow("Payload bytes scanned", scannedBytes.toLocaleString());
    addRow("BF16 values checked", scannedValues.toLocaleString());
    addRow("Read chunks", String(chunks));
    addRow("Non-finite values", "0");
    addRow("Elapsed time", ((performance.now() - started) / 1000).toFixed(2) + " s");
    addRow("Network upload", "None · all bytes read from the local File object");

    setStatus("All " + scannedValues.toLocaleString() +
      " local BF16 payload values across " + tensors.length +
      " tensors passed the finite-value and byte-range checks.", "success");
  } catch (error) {
    console.error("[Illustrious] Full local LoRA payload check failed:", error);
    setStatus("Full local LoRA payload check failed: " + (error?.message || String(error)), "error");
  } finally {
    button.disabled = false;
  }
});
