const MODEL_BASE = "https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/resolve/main/";
const MANIFEST_URL = MODEL_BASE + "manifest.json";

const button = document.querySelector("#inspectUnetMetadata");
const status = document.querySelector("#unetMetadataStatus");
const output = document.querySelector("#unetMetadataOutput");

function setStatus(message, type = "info") {
  status.textContent = message;
  status.dataset.type = type;
}

function readVarint(bytes, offset) {
  let value = 0n;
  let shift = 0n;
  let position = offset;

  while (position < bytes.length && shift <= 63n) {
    const byte = bytes[position++];
    value |= BigInt(byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) return { value, next: position };
    shift += 7n;
  }

  throw new Error("Malformed or oversized protobuf varint.");
}

function parseFields(bytes) {
  const fields = [];
  let offset = 0;

  while (offset < bytes.length) {
    const tag = readVarint(bytes, offset);
    offset = tag.next;
    const fieldNumber = Number(tag.value >> 3n);
    const wireType = Number(tag.value & 7n);
    if (!fieldNumber) throw new Error("Invalid protobuf field number 0.");

    if (wireType === 0) {
      const item = readVarint(bytes, offset);
      fields.push({ number: fieldNumber, wire: wireType, value: item.value });
      offset = item.next;
    } else if (wireType === 1) {
      if (offset + 8 > bytes.length) throw new Error("Truncated fixed64 protobuf field.");
      fields.push({ number: fieldNumber, wire: wireType, value: bytes.subarray(offset, offset + 8) });
      offset += 8;
    } else if (wireType === 2) {
      const length = readVarint(bytes, offset);
      offset = length.next;
      const size = Number(length.value);
      if (!Number.isSafeInteger(size) || size < 0 || offset + size > bytes.length) {
        throw new Error("Invalid protobuf length-delimited field.");
      }
      fields.push({ number: fieldNumber, wire: wireType, value: bytes.subarray(offset, offset + size) });
      offset += size;
    } else if (wireType === 5) {
      if (offset + 4 > bytes.length) throw new Error("Truncated fixed32 protobuf field.");
      fields.push({ number: fieldNumber, wire: wireType, value: bytes.subarray(offset, offset + 4) });
      offset += 4;
    } else {
      throw new Error("Unsupported protobuf wire type " + wireType + ".");
    }
  }

  return fields;
}

function stringValue(bytes) {
  return new TextDecoder().decode(bytes);
}

function fieldsOf(fields, number, wire = 2) {
  return fields.filter(field => field.number === number && field.wire === wire);
}

const ELEMENT_TYPES = {
  1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
  6: "int32", 7: "int64", 9: "bool", 10: "float16", 11: "float64",
  12: "uint32", 13: "uint64", 14: "complex64", 15: "complex128",
  16: "bfloat16", 17: "float8e4m3fn", 18: "float8e4m3fnuz",
  19: "float8e5m2", 20: "float8e5m2fnuz"
};

function parseShape(bytes) {
  const fields = parseFields(bytes);
  return fieldsOf(fields, 1).map(item => {
    const dim = parseFields(item.value);
    const numeric = dim.find(field => field.number === 1 && field.wire === 0);
    const symbolic = dim.find(field => field.number === 2 && field.wire === 2);
    if (numeric) return numeric.value.toString();
    if (symbolic) return stringValue(symbolic.value);
    return "?";
  });
}

function parseValueInfo(bytes) {
  const fields = parseFields(bytes);
  const name = fields.find(field => field.number === 1 && field.wire === 2);
  const type = fields.find(field => field.number === 2 && field.wire === 2);
  let dtype = "unknown";
  let shape = [];

  if (type) {
    const typeFields = parseFields(type.value);
    const tensor = typeFields.find(field => field.number === 1 && field.wire === 2);
    if (tensor) {
      const tensorFields = parseFields(tensor.value);
      const elem = tensorFields.find(field => field.number === 1 && field.wire === 0);
      const dimensions = tensorFields.find(field => field.number === 2 && field.wire === 2);
      if (elem) dtype = ELEMENT_TYPES[Number(elem.value)] || "elem_type_" + elem.value;
      if (dimensions) shape = parseShape(dimensions.value);
    }
  }

  return { name: name ? stringValue(name.value) : "(unnamed)", dtype, shape };
}

function parseInitializer(bytes) {
  const fields = parseFields(bytes);
  const nameField = fields.find(field => field.number === 8 && field.wire === 2);
  const typeField = fields.find(field => field.number === 2 && field.wire === 0);
  const dimensions = fieldsOf(fields, 1, 0).map(field => field.value.toString());
  const external = fieldsOf(fields, 13).map(field => {
    const entry = parseFields(field.value);
    const key = entry.find(item => item.number === 1 && item.wire === 2);
    const value = entry.find(item => item.number === 2 && item.wire === 2);
    return [key ? stringValue(key.value) : "?", value ? stringValue(value.value) : "?"];
  });
  return {
    name: nameField ? stringValue(nameField.value) : "(unnamed)",
    dtype: typeField ? (ELEMENT_TYPES[Number(typeField.value)] || String(typeField.value)) : "unknown",
    shape: dimensions,
    external
  };
}

function inspectOnnxGraph(buffer) {
  const modelFields = parseFields(new Uint8Array(buffer));
  const graphField = modelFields.find(field => field.number === 7 && field.wire === 2);
  if (!graphField) throw new Error("ONNX ModelProto has no graph field.");

  const graphFields = parseFields(graphField.value);
  const graphName = graphFields.find(field => field.number === 2 && field.wire === 2);
  const inputs = fieldsOf(graphFields, 11).map(field => parseValueInfo(field.value));
  const outputs = fieldsOf(graphFields, 12).map(field => parseValueInfo(field.value));
  const nodes = fieldsOf(graphFields, 1).length;
  const initializers = fieldsOf(graphFields, 5).map(field => parseInitializer(field.value));
  const external = initializers.filter(item => item.external.length);

  return {
    graphName: graphName ? stringValue(graphName.value) : "(unnamed)",
    nodes,
    inputs,
    outputs,
    initializerCount: initializers.length,
    externalInitializerCount: external.length,
    initializerExamples: external.slice(0, 3)
  };
}

function sha256(bytes) {
  return crypto.subtle.digest("SHA-256", bytes).then(hash =>
    Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, "0")).join("")
  );
}

function compactIndex(data) {
  const rootKeys = data && typeof data === "object" ? Object.keys(data) : [];
  let mapName = "(root)";
  let map = data;

  if (data && typeof data === "object" && !Array.isArray(data)) {
    for (const [key, value] of Object.entries(data)) {
      if (value && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length > 20) {
        mapName = key;
        map = value;
        break;
      }
    }
  }

  const entries = Array.isArray(map)
    ? map.slice(0, 8).map((value, index) => [String(index), value])
    : map && typeof map === "object"
      ? Object.entries(map).slice(0, 8)
      : [];

  return {
    rootType: Array.isArray(data) ? "array" : typeof data,
    rootKeys: rootKeys.slice(0, 16),
    mapName,
    mapEntryCount: map && typeof map === "object" ? Object.keys(map).length : 0,
    samples: entries.map(([key, value]) => ({
      name: key,
      value: JSON.stringify(value)?.slice(0, 260) ?? String(value)
    }))
  };
}

function formatGraph(graph) {
  const lines = [
    "Graph: " + graph.graphName,
    "Nodes: " + graph.nodes.toLocaleString(),
    "Initializers: " + graph.initializerCount.toLocaleString() +
      " (" + graph.externalInitializerCount.toLocaleString() + " external)",
    "Inputs:"
  ];

  for (const item of graph.inputs) {
    lines.push("  " + item.name + " · " + item.dtype + " · [" + item.shape.join(", ") + "]");
  }

  lines.push("Outputs:");
  for (const item of graph.outputs) {
    lines.push("  " + item.name + " · " + item.dtype + " · [" + item.shape.join(", ") + "]");
  }

  if (graph.initializerExamples.length) {
    lines.push("External initializer examples:");
    for (const item of graph.initializerExamples) {
      lines.push("  " + item.name + " · " + item.dtype + " · [" + item.shape.join(", ") + "]");
      for (const [key, value] of item.external) lines.push("    " + key + ": " + value);
    }
  }

  return lines.join("\n");
}

button.addEventListener("click", async () => {
  button.disabled = true;
  output.textContent = "";
  output.hidden = false;

  try {
    setStatus("Fetching the small manifest, three ONNX graph files, and three weight indexes. No multi-gigabyte tensor shards are requested…");
    const manifestResponse = await fetch(MANIFEST_URL, { credentials: "omit", cache: "no-cache" });
    if (!manifestResponse.ok) throw new Error("Manifest request failed: HTTP " + manifestResponse.status);
    const manifest = await manifestResponse.json();

    const components = ["unet_a", "unet_b", "unet_c"];
    const files = [];
    for (const name of components) {
      const component = manifest?.components?.[name];
      if (!component?.onnx || !component?.weightsIndex) {
        throw new Error("Manifest is missing graph or weightsIndex for " + name + ".");
      }
      files.push(component.onnx, component.weightsIndex);
    }

    const fileMap = new Map((manifest.files || []).map(file => [file.path, file]));
    const started = performance.now();
    const downloaded = await Promise.all(files.map(async path => {
      const expected = fileMap.get(path);
      if (!expected?.sha256) throw new Error("Manifest has no SHA-256 record for " + path + ".");
      const response = await fetch(MODEL_BASE + path, { credentials: "omit", cache: "force-cache" });
      if (!response.ok) throw new Error("Request failed for " + path + ": HTTP " + response.status);
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength !== expected.size) {
        throw new Error(path + " size mismatch: expected " + expected.size + ", received " + buffer.byteLength + ".");
      }
      const actualHash = await sha256(buffer);
      if (actualHash !== expected.sha256) throw new Error(path + " SHA-256 mismatch.");
      return { path, buffer, size: buffer.byteLength };
    }));

    const byPath = new Map(downloaded.map(file => [file.path, file]));
    const reports = [];
    let verifiedBytes = 0;

    for (const name of components) {
      const component = manifest.components[name];
      const graphFile = byPath.get(component.onnx);
      const indexFile = byPath.get(component.weightsIndex);
      verifiedBytes += graphFile.size + indexFile.size;

      const graph = inspectOnnxGraph(graphFile.buffer);
      const index = JSON.parse(new TextDecoder().decode(indexFile.buffer));
      reports.push(
        name.toUpperCase() + " · " + graphFile.size.toLocaleString() + " byte graph · " +
          indexFile.size.toLocaleString() + " byte index\n" +
          formatGraph(graph) + "\n" +
          "Weight index:\n" + JSON.stringify(compactIndex(index), null, 2)
      );
    }

    output.textContent =
      "Manifest: " + manifest.packageId + "\n" +
      "Verified metadata files: " + downloaded.length + " / " + downloaded.length +
      " (all sizes and SHA-256 hashes match)\n" +
      "Metadata downloaded: " + (verifiedBytes / 1024 / 1024).toFixed(2) + " MiB\n" +
      "UNet tensor shards downloaded: 0\n" +
      "Inspection time: " + ((performance.now() - started) / 1000).toFixed(1) + " s\n\n" +
      reports.join("\n\n" + "─".repeat(58) + "\n\n");

    setStatus("All three split-UNet graph files and weight indexes were fetched and hash-verified. No multi-gigabyte weight shards were downloaded.", "success");
  } catch (error) {
    console.error("[Illustrious] UNet metadata inspection failed:", error);
    setStatus("UNet metadata inspection failed: " + (error?.message || String(error)), "error");
  } finally {
    button.disabled = false;
  }
});
