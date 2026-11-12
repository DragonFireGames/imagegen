import { CLIPTokenizer } from "../vendor/transformers/transformers.min.js";
import { acquireHeavyModelTest, releaseHeavyModelTest } from "./heavy-model-test-lock.js";
import { prepareLoraComponent, releaseLoraPayload } from "./lora-patcher.js";
import { fetchModelResource, prepareModelCache, exportModelBundle, importModelBundle } from "./model-cache.js";

const MODEL_ID = "aaaaaeeeee/blip-sdxl-anime-wai-illustrious17";
const MODEL_BASE = "https://huggingface.co/" + MODEL_ID + "/resolve/main/";
const MANIFEST_URL = MODEL_BASE + "manifest.json";
const SCHEDULER_URL = "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/b41f1c324e32e03f9e104bd9ace0992e4e135597/scheduler/scheduler_config.json";
const VAE_SCALING_FACTOR = 0.18215; // The WAI pack's Kohaku XL beta5 base VAE config.

const runtimePreference = new URLSearchParams(window.location.search).get("runtime") || "auto";
const graphCaptureEnabled = new URLSearchParams(window.location.search).get("graphCapture") === "1";
const keepVaeResident = new URLSearchParams(window.location.search).get("keepVae") === "1";
const preferredUnetLayout = new URLSearchParams(window.location.search).get("layout") === "NHWC" ? "NHWC" : null;
const requestedUnetValidationMode = new URLSearchParams(window.location.search).get("validation");
const preferredUnetValidationMode = ["basic", "wgpuOnly", "full"].includes(requestedUnetValidationMode)
  ? requestedUnetValidationMode
  : null;
const LOCAL_ORT_BASE = new URL("../vendor/onnxruntime-web/", import.meta.url).href;

async function loadOrtRuntime() {
  const jspiSupported = typeof WebAssembly.Suspending === "function" &&
    typeof WebAssembly.promising === "function";
  if (runtimePreference !== "asyncify" && jspiSupported) {
    try {
      const runtime = await import("../vendor/onnxruntime-web/ort.jspi.bundle.min.mjs");
      runtime.env.wasm.wasmPaths = LOCAL_ORT_BASE;
      runtime.env.wasm.numThreads = 0;
      runtime.env.logLevel = "warning";
      return {
        ort: runtime,
        label: "ONNX Runtime Web 1.31.1 · native WebGPU · JSPI",
        note: ""
      };
    } catch (error) {
      const fallbackReason = error?.message || String(error);
      const runtime = await import("../vendor/onnxruntime-web/ort.webgpu.min.mjs");
      runtime.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
      runtime.env.wasm.numThreads = 0;
      runtime.env.logLevel = "warning";
      return {
        ort: runtime,
        label: "ONNX Runtime Web 1.31.1 · native WebGPU · Asyncify fallback",
        note: "JSPI runtime import failed; using the locally bundled Asyncify runtime (" + fallbackReason + ")."
      };
    }
  }
  const runtime = await import("../vendor/onnxruntime-web/ort.webgpu.min.mjs");
  runtime.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
  runtime.env.wasm.numThreads = 0;
  runtime.env.logLevel = "warning";
  return {
    ort: runtime,
    label: "ONNX Runtime Web 1.31.1 · native WebGPU · Asyncify",
    note: runtimePreference === "asyncify"
      ? "Local Asyncify runtime explicitly selected with ?runtime=asyncify."
      : "This browser does not expose WebAssembly JSPI; using the local Asyncify runtime."
  };
}

const { ort, label: runtimeLabel, note: runtimeNote } = await loadOrtRuntime();
// Select the discrete/high-performance GPU where the browser exposes a choice.
// This must be set before ORT creates its first WebGPU session.
if (ort.env.webgpu) ort.env.webgpu.powerPreference = "high-performance";

// Keep the legacy diagnostic switch recognizable, but do not enable the JSEP-only
// profiling callback on this app's native WebGPU execution-provider path.
const gpuProfilingRequested = new URLSearchParams(window.location.search).get("gpuProfile") === "1";

const button = document.querySelector("#generateImage");
const cancelButton = document.querySelector("#cancelGeneration");
const status = document.querySelector("#generationStatus");
const progress = document.querySelector("#generationProgress");
const progressLabel = document.querySelector("#generationProgressLabel");
const logOutput = document.querySelector("#generationLog");
const generatedResults = document.querySelector("#generatedResults");
const prepareEngineButton = document.querySelector("#prepareInferenceEngine");
const releaseEngineButton = document.querySelector("#releaseInferenceEngine");

const fields = {
  prompt: document.querySelector("#generationPrompt"),
  negative: document.querySelector("#generationNegativePrompt"),
  suffix: document.querySelector("#generationSuffix"),
  width: document.querySelector("#generationWidth"),
  height: document.querySelector("#generationHeight"),
  steps: document.querySelector("#generationSteps"),
  cfg: document.querySelector("#generationCfg"),
  seed: document.querySelector("#generationSeed"),
  batch: document.querySelector("#generationBatch"),
  parallel: document.querySelector("#generationParallel"),
  lora: document.querySelector("#generationLoraEnabled"),
  loraStrength: document.querySelector("#generationLoraStrength"),
  loraFile: document.querySelector("#loraFile")
};

const adapterFileLabel = document.querySelector("#fileName");
let expectedAdapterFile = null;

function adapterFileMatches(file, expected) {
  return !expected || Boolean(file && file.name === expected.name && file.size === expected.sizeBytes);
}

function updateAdapterFileLabel() {
  if (!adapterFileLabel) return;
  const file = fields.loraFile.files?.[0] || null;
  if (expectedAdapterFile && !adapterFileMatches(file, expectedAdapterFile)) {
    adapterFileLabel.textContent = "Select " + expectedAdapterFile.name + " · " +
      formatBytes(expectedAdapterFile.sizeBytes) + " to match imported settings";
    return;
  }
  adapterFileLabel.textContent = file
    ? file.name + " · " + formatBytes(file.size)
    : "No adapter selected";
}

fields.loraFile?.addEventListener("change", updateAdapterFileLabel);

let cancelRequested = false;
let generationStarted = 0;
let generatedImageUrls = [];
let activeBatch = null;
let residentUnetSessions = null;
let residentUnetKey = null;
let residentVaeSession = null;
let residentEngineLock = false;
const conditioningCache = new Map();

function setStatus(message, type = "info") {
  status.textContent = activeBatch
    ? (activeBatch.span > 1
      ? "Images " + (activeBatch.index + 1) + "–" + (activeBatch.index + activeBatch.span) + "/" + activeBatch.count
      : "Image " + (activeBatch.index + 1) + "/" + activeBatch.count) + " · " + message
    : message;
  status.dataset.type = type;
}

function setProgress(value, label) {
  progress.value = activeBatch
    ? ((activeBatch.index + Math.max(0, Math.min(100, value)) / 100 * (activeBatch.span || 1)) / activeBatch.count) * 100
    : value;
  progressLabel.textContent = activeBatch
    ? "Batch " + (activeBatch.index + 1) + "/" + activeBatch.count + " · " + label
    : label;
}

function clearGeneratedResults() {
  for (const url of generatedImageUrls) URL.revokeObjectURL(url);
  generatedImageUrls = [];
  generatedResults?.replaceChildren();
}

function appendGeneratedResult(blob, options, index, count) {
  if (!generatedResults) throw new Error("The generated-results gallery is missing.");
  const url = URL.createObjectURL(blob);
  generatedImageUrls.push(url);

  const item = document.createElement("article");
  item.className = "generated-item";
  const preview = document.createElement("img");
  preview.src = url;
  preview.alt = "Generated image " + (index + 1);
  preview.loading = "lazy";

  const metadata = document.createElement("p");
  metadata.textContent = "Image " + (index + 1) + "/" + count + " · " +
    options.width + "×" + options.height + " · seed " + options.seed;

  const download = document.createElement("a");
  download.href = url;
  download.download = "illustrious-seed-" + options.seed + ".png";
  download.textContent = "Download PNG · seed " + options.seed;

  item.append(preview, metadata, download);
  generatedResults.append(item);
}

function log(message) {
  const elapsed = ((performance.now() - generationStarted) / 1000).toFixed(2);
  const line = "[" + elapsed.padStart(8) + "s] " + message;
  logOutput.textContent += line + "\n";
  logOutput.scrollTop = logOutput.scrollHeight;
  console.log("[Illustrious generation] " + line);
}

function yieldFrames() {
  return new Promise(resolve => requestAnimationFrame(() =>
    requestAnimationFrame(resolve)
  ));
}

function checkCancelled() {
  if (cancelRequested) throw new Error("Generation cancelled by user.");
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "unknown";
  if (bytes >= 1024 ** 3) return (bytes / 1024 ** 3).toFixed(2) + " GiB";
  return (bytes / 1024 ** 2).toFixed(1) + " MiB";
}

async function fetchJSON(url, label, forceRefresh = false) {
  const response = await fetchModelResource(url, {
    credentials: "omit",
    forceRefresh,
    onCacheMiss: path => log("MODEL CACHE MISS · " + path + " · using a visible fallback download.")
  });
  if (!response.ok) throw new Error(label + " request failed: HTTP " + response.status);
  return response.json();
}

const tokenizerCache = new Map();

async function loadTokenizer(subfolder) {
  if (tokenizerCache.has(subfolder)) return tokenizerCache.get(subfolder);
  const folderURL = MODEL_BASE + subfolder + "/";

  async function readJSON(filename) {
    const url = folderURL + filename;
    const response = await fetchModelResource(url, {
      credentials: "omit",
      onCacheMiss: path => log("MODEL CACHE MISS · " + path + " · using a visible fallback download.")
    });
    if (!response.ok) throw new Error("Tokenizer request failed: " + filename + " · HTTP " + response.status);
    return response.json();
  }

  async function readText(filename) {
    const url = folderURL + filename;
    const response = await fetchModelResource(url, {
      credentials: "omit",
      onCacheMiss: path => log("MODEL CACHE MISS · " + path + " · using a visible fallback download.")
    });
    if (!response.ok) throw new Error("Tokenizer request failed: " + filename + " · HTTP " + response.status);
    return response.text();
  }

  const [vocab, mergesText, specialMap] = await Promise.all([
    readJSON("vocab.json"),
    readText("merges.txt"),
    readJSON("special_tokens_map.json")
  ]);

  function getToken(value, name) {
    const token = typeof value === "string" ? value : value?.content;
    if (typeof token !== "string" || !token.length) {
      throw new Error(subfolder + "/special_tokens_map.json is missing " + name + ".");
    }
    return token;
  }

  const bosToken = getToken(specialMap.bos_token, "bos_token");
  const eosToken = getToken(specialMap.eos_token, "eos_token");
  const padToken = getToken(specialMap.pad_token, "pad_token");
  const unkToken = getToken(specialMap.unk_token, "unk_token");
  const bosId = vocab[bosToken];
  const eosId = vocab[eosToken];
  const padId = vocab[padToken];
  if (!Number.isSafeInteger(bosId) || !Number.isSafeInteger(eosId) ||
      bosId !== 49406 || eosId !== 49407) {
    throw new Error(subfolder + " has unexpected CLIP BOS/EOS IDs: " + bosId + " / " + eosId + ".");
  }
  if (!Number.isSafeInteger(padId) || !Number.isSafeInteger(vocab[unkToken])) {
    throw new Error(subfolder + " is missing its configured pad/unknown token.");
  }

  function makeAddedToken(value, id, content) {
    return {
      id,
      content,
      single_word: value?.single_word ?? false,
      lstrip: value?.lstrip ?? false,
      rstrip: value?.rstrip ?? false,
      normalized: value?.normalized ?? false,
      special: true
    };
  }

  const merges = mergesText.split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line && !line.startsWith("#"))
    .map((line, index) => {
      const pair = line.split(/\s+/);
      if (pair.length !== 2) throw new Error(subfolder + "/merges.txt has an invalid pair on line " + (index + 2) + ".");
      return pair;
    });
  if (!merges.length) throw new Error(subfolder + "/merges.txt contains no BPE merges.");

  const tokenizerJSON = {
    version: "1.0",
    truncation: null,
    padding: null,
    added_tokens: [
      makeAddedToken(specialMap.bos_token, bosId, bosToken),
      makeAddedToken(specialMap.eos_token, eosId, eosToken)
    ],
    normalizer: {
      type: "Sequence",
      normalizers: [
        { type: "NFC" },
        { type: "Replace", pattern: { Regex: "\\s+" }, content: " " },
        { type: "Lowercase" }
      ]
    },
    pre_tokenizer: {
      type: "Sequence",
      pretokenizers: [
        {
          type: "Split",
          pattern: {
            Regex: "'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"
          },
          behavior: "Removed",
          invert: true
        },
        { type: "ByteLevel", add_prefix_space: false, trim_offsets: true }
      ]
    },
    post_processor: {
      type: "RobertaProcessing",
      sep: [eosToken, eosId],
      cls: [bosToken, bosId],
      trim_offsets: false,
      add_prefix_space: false
    },
    decoder: { type: "ByteLevel", add_prefix_space: true, trim_offsets: true },
    model: {
      type: "BPE",
      dropout: null,
      unk_token: unkToken,
      continuing_subword_prefix: "",
      end_of_word_suffix: "</w>",
      fuse_unk: false,
      vocab,
      merges
    }
  };
  const tokenizerConfig = {
    tokenizer_class: "CLIPTokenizer",
    model_max_length: 77,
    bos_token: bosToken,
    eos_token: eosToken,
    pad_token: padToken,
    unk_token: unkToken,
    clean_up_tokenization_spaces: false
  };
  const tokenizer = new CLIPTokenizer(tokenizerJSON, tokenizerConfig);
  if (tokenizer.bos_token_id !== bosId || tokenizer.eos_token_id !== eosId) {
    throw new Error(subfolder + " CLIP tokenizer special-token IDs did not initialize correctly.");
  }
  tokenizerCache.set(subfolder, tokenizer);
  return tokenizer;
}

function tokenIds(encoded, label) {
  const source = encoded?.input_ids;
  if (!source?.data || !Array.isArray(source.dims) ||
      source.dims.length !== 2 || source.dims[0] !== 1 || source.dims[1] !== 77) {
    throw new Error(label + " tokenizer did not return input_ids [1, 77].");
  }
  const output = new BigInt64Array(77);
  for (let i = 0; i < output.length; i++) {
    const id = Number(source.data[i]);
    if (!Number.isSafeInteger(id)) throw new Error(label + " contains an invalid token ID.");
    output[i] = BigInt(id);
  }
  return output;
}

function combineHidden(hiddenL, hiddenG, label) {
  const combined = new Float16Array(77 * 2048);
  for (let token = 0; token < 77; token++) {
    combined.set(hiddenL.subarray(token * 768, token * 768 + 768), token * 2048);
    combined.set(hiddenG.subarray(token * 1280, token * 1280 + 1280), token * 2048 + 768);
  }
  for (let i = 0; i < combined.length; i++) {
    if (!Number.isFinite(combined[i])) throw new Error(label + " contains a non-finite value at " + i + ".");
  }
  return combined;
}

function fp16Values(tensor, label, shape) {
  if (!tensor?.data || tensor.type !== "float16" || !Array.isArray(tensor.dims)) {
    throw new Error(label + " is not a readable FP16 tensor.");
  }
  if (tensor.dims.length !== shape.length ||
      tensor.dims.some((dimension, index) => dimension !== shape[index])) {
    throw new Error(label + " returned shape [" + tensor.dims.join(", ") +
      "], expected [" + shape.join(", ") + "].");
  }
  let values;
  if (tensor.data instanceof Float16Array) {
    values = tensor.data;
  } else if (tensor.data instanceof Uint16Array) {
    const buffer = tensor.data.buffer.slice(
      tensor.data.byteOffset,
      tensor.data.byteOffset + tensor.data.byteLength
    );
    values = new Float16Array(buffer);
  } else {
    throw new Error(label + " backing data is " + tensor.data.constructor.name + "; expected Float16Array/Uint16Array.");
  }
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) throw new Error(label + " contains a non-finite value at " + i + ".");
  }
  return new Float16Array(values);
}

function imageTensorValues(tensor, label) {
  if (!tensor?.data || !Array.isArray(tensor.dims)) throw new Error(label + " has no readable tensor data.");
  if (tensor.type === "float16") {
    if (tensor.data instanceof Float16Array) return tensor.data;
    if (tensor.data instanceof Uint16Array) {
      return new Float16Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);
    }
  } else if (tensor.type === "float32" && tensor.data instanceof Float32Array) {
    return tensor.data;
  }
  throw new Error(label + " has unsupported backing data " + tensor.data.constructor.name +
    " for dtype " + tensor.type + ".");
}

async function externalData(component, manifest) {
  if (!component?.onnx || !Array.isArray(component.shards) || !component.shards.length) {
    throw new Error("Model manifest component is missing its ONNX graph or external-data shards.");
  }
  const records = new Map((manifest?.files || []).map(item => [item.path, item]));
  return Promise.all(component.shards.map(async path => {
    const record = records.get(path);
    const response = await fetchModelResource(MODEL_BASE + path, {
      credentials: "omit",
      expectedSize: record?.size,
      onCacheMiss: missing => log("MODEL CACHE MISS · " + missing + " · downloading a visible fallback copy.")
    });
    if (!response.ok) throw new Error("External weight shard request failed: " + path + " · HTTP " + response.status);
    return {
      path: path.slice(path.lastIndexOf("/") + 1),
      data: await response.blob()
    };
  }));
}

async function loadSession(
  manifest,
  key,
  label,
  patchedExternalData = null,
  preferredOutputLocation = null,
  enableGraphCapture = false,
  preferredLayout = null,
  validationMode = null
) {
  const component = manifest?.components?.[key];
  const started = performance.now();
  log("Loading " + label + " (" + formatBytes(component?.bytes) + ")…");
  const graphRecord = (manifest.files || []).find(item => item.path === component.onnx);
  const graphResponse = await fetchModelResource(MODEL_BASE + component.onnx, {
    credentials: "omit",
    expectedSize: graphRecord?.size,
    onCacheMiss: path => log("MODEL CACHE MISS · " + path + " · downloading a visible fallback copy.")
  });
  if (!graphResponse.ok) throw new Error(label + " ONNX graph request failed · HTTP " + graphResponse.status);
  const graphData = await graphResponse.arrayBuffer();
  const weightData = patchedExternalData || await externalData(component, manifest);
  const webgpuProvider = { name: "webgpu" };
  if (preferredLayout) webgpuProvider.preferredLayout = preferredLayout;
  if (validationMode) webgpuProvider.validationMode = validationMode;
  const sessionOptions = {
    executionProviders: preferredLayout || validationMode
      ? [webgpuProvider]
      : ["webgpu"],
    graphOptimizationLevel: "all",
    externalData: weightData
  };
  if (preferredOutputLocation) sessionOptions.preferredOutputLocation = preferredOutputLocation;
  if (enableGraphCapture) sessionOptions.enableGraphCapture = true;
  const session = await ort.InferenceSession.create(graphData, sessionOptions);
  log(label + " session created in " + ((performance.now() - started) / 1000).toFixed(2) +
    "s · inputs: " + session.inputNames.join(", ") +
    (preferredOutputLocation ? " · GPU-resident outputs enabled" : "") +
    (enableGraphCapture ? " · WebGPU graph capture enabled" : "") +
    (preferredLayout ? " · preferred layout " + preferredLayout : ""));
  return session;
}

function canBatchEncoder(session, inputName) {
  const index = session.inputNames.indexOf(inputName);
  const firstDimension = session.inputMetadata?.[index]?.shape?.[0];
  return typeof firstDimension !== "number" || firstDimension < 0 || firstDimension >= 2;
}

function combineTokenBatches(positive, negative, label) {
  if (positive.length !== 77 || negative.length !== 77) {
    throw new Error(label + " tokenizers must return exactly 77 IDs per prompt.");
  }
  const batch = new BigInt64Array(154);
  batch.set(positive, 0);
  batch.set(negative, 77);
  return batch;
}

function splitFp16Batch(values, sampleSize, batchIndex) {
  return new Float16Array(values.subarray(batchIndex * sampleSize, (batchIndex + 1) * sampleSize));
}

async function encodePrompts(manifest, prompt, negativePrompt, lora, cfg) {
  const options = { padding: "max_length", truncation: true, max_length: 77 };
  const branchIndex = cfg === 0 ? 0 : cfg === 1 ? 1 : null;
  const encodePositive = branchIndex !== 0;
  const encodeNegative = branchIndex !== 1;
  const branchDescription = branchIndex === 1
    ? "positive prompt only · CFG 1"
    : branchIndex === 0
      ? "negative prompt only · CFG 0"
      : "positive and negative prompts";
  log("Loading CLIP-L and tokenizing " + branchDescription + "…");
  const tokenizerL = await loadTokenizer("tokenizer");
  const encodedL = encodePositive ? await tokenizerL(prompt, options) : null;
  const negativeEncodedL = encodeNegative ? await tokenizerL(negativePrompt, options) : null;
  const idsL = encodedL ? tokenIds(encodedL, "CLIP-L positive") : null;
  const negativeIdsL = negativeEncodedL ? tokenIds(negativeEncodedL, "CLIP-L negative") : null;
  const startedL = performance.now();
  let sessionL;
  let hiddenL;
  let negativeHiddenL;
  try {
    const patchedL = lora
      ? await prepareLoraComponent(lora.file, manifest, "text_encoder", lora.strength, log, checkCancelled)
      : null;
    sessionL = await loadSession(manifest, "text_encoder", "CLIP-L", patchedL?.externalData);
    if (branchIndex !== null) {
      const selectedIds = branchIndex === 1 ? idsL : negativeIdsL;
      const output = await sessionL.run({ input_ids: new ort.Tensor("int64", selectedIds, [1, 77]) });
      const hidden = fp16Values(output.penultimate_hidden_state, "CLIP-L selected hidden", [1, 77, 768]);
      if (branchIndex === 1) hiddenL = hidden;
      else negativeHiddenL = hidden;
      log("CLIP-L encoded only the selected CFG branch.");
    } else if (canBatchEncoder(sessionL, "input_ids")) {
      const outputs = await sessionL.run({
        input_ids: new ort.Tensor("int64", combineTokenBatches(idsL, negativeIdsL, "CLIP-L"), [2, 77])
      });
      const combined = fp16Values(outputs.penultimate_hidden_state, "CLIP-L positive/negative hidden", [2, 77, 768]);
      hiddenL = splitFp16Batch(combined, 77 * 768, 0);
      negativeHiddenL = splitFp16Batch(combined, 77 * 768, 1);
      log("CLIP-L encoded both prompts in one batch.");
    } else {
      const positive = await sessionL.run({ input_ids: new ort.Tensor("int64", idsL, [1, 77]) });
      hiddenL = fp16Values(positive.penultimate_hidden_state, "CLIP-L positive hidden", [1, 77, 768]);
      const negative = await sessionL.run({ input_ids: new ort.Tensor("int64", negativeIdsL, [1, 77]) });
      negativeHiddenL = fp16Values(negative.penultimate_hidden_state, "CLIP-L negative hidden", [1, 77, 768]);
      log("CLIP-L graph has a fixed batch of one; used two compatible calls.");
    }
  } finally {
    if (sessionL) await sessionL.release();
  }
  await yieldFrames();
  log("CLIP-L " + branchDescription + " encoded in " + ((performance.now() - startedL) / 1000).toFixed(2) + "s; released.");

  log("Loading CLIP-bigG and tokenizing " + branchDescription + "…");
  const tokenizerG = await loadTokenizer("tokenizer_2");
  const encodedG = encodePositive ? await tokenizerG(prompt, options) : null;
  const negativeEncodedG = encodeNegative ? await tokenizerG(negativePrompt, options) : null;
  const idsG = encodedG ? tokenIds(encodedG, "CLIP-bigG positive") : null;
  const negativeIdsG = negativeEncodedG ? tokenIds(negativeEncodedG, "CLIP-bigG negative") : null;
  const startedG = performance.now();
  let sessionG;
  let hiddenG;
  let negativeHiddenG;
  let pooledG;
  let negativePooledG;
  try {
    const patchedG = lora
      ? await prepareLoraComponent(lora.file, manifest, "text_encoder_2", lora.strength, log, checkCancelled)
      : null;
    sessionG = await loadSession(manifest, "text_encoder_2", "CLIP-bigG", patchedG?.externalData);
    if (branchIndex !== null) {
      const selectedIds = branchIndex === 1 ? idsG : negativeIdsG;
      const output = await sessionG.run({ input_ids: new ort.Tensor("int64", selectedIds, [1, 77]) });
      const hidden = fp16Values(output.penultimate_hidden_state, "CLIP-bigG selected hidden", [1, 77, 1280]);
      const pooled = fp16Values(output.text_embeds, "CLIP-bigG selected pooled", [1, 1280]);
      if (branchIndex === 1) {
        hiddenG = hidden;
        pooledG = pooled;
      } else {
        negativeHiddenG = hidden;
        negativePooledG = pooled;
      }
      log("CLIP-bigG encoded only the selected CFG branch.");
    } else if (canBatchEncoder(sessionG, "input_ids")) {
      const outputs = await sessionG.run({
        input_ids: new ort.Tensor("int64", combineTokenBatches(idsG, negativeIdsG, "CLIP-bigG"), [2, 77])
      });
      const combinedHidden = fp16Values(outputs.penultimate_hidden_state, "CLIP-bigG positive/negative hidden", [2, 77, 1280]);
      const combinedPooled = fp16Values(outputs.text_embeds, "CLIP-bigG positive/negative pooled", [2, 1280]);
      hiddenG = splitFp16Batch(combinedHidden, 77 * 1280, 0);
      negativeHiddenG = splitFp16Batch(combinedHidden, 77 * 1280, 1);
      pooledG = splitFp16Batch(combinedPooled, 1280, 0);
      negativePooledG = splitFp16Batch(combinedPooled, 1280, 1);
      log("CLIP-bigG encoded both prompts in one batch.");
    } else {
      const positive = await sessionG.run({ input_ids: new ort.Tensor("int64", idsG, [1, 77]) });
      hiddenG = fp16Values(positive.penultimate_hidden_state, "CLIP-bigG positive hidden", [1, 77, 1280]);
      pooledG = fp16Values(positive.text_embeds, "CLIP-bigG positive pooled", [1, 1280]);
      const negative = await sessionG.run({ input_ids: new ort.Tensor("int64", negativeIdsG, [1, 77]) });
      negativeHiddenG = fp16Values(negative.penultimate_hidden_state, "CLIP-bigG negative hidden", [1, 77, 1280]);
      negativePooledG = fp16Values(negative.text_embeds, "CLIP-bigG negative pooled", [1, 1280]);
      log("CLIP-bigG graph has a fixed batch of one; used two compatible calls.");
    }
  } finally {
    if (sessionG) await sessionG.release();
  }
  await yieldFrames();
  log("CLIP-bigG " + branchDescription + " encoded in " + ((performance.now() - startedG) / 1000).toFixed(2) + "s; released.");

  const positive = encodePositive ? combineHidden(hiddenL, hiddenG, "Positive conditioning") : null;
  const negative = encodeNegative ? combineHidden(negativeHiddenL, negativeHiddenG, "Negative conditioning") : null;
  const hiddenLength = 77 * 2048;
  const pooledLength = 1280;
  const encoderHidden = new Float16Array(2 * hiddenLength);
  const textEmbeds = new Float16Array(2 * pooledLength);
  if (negative) {
    encoderHidden.set(negative, 0);
    textEmbeds.set(negativePooledG, 0);
  }
  if (positive) {
    encoderHidden.set(positive, hiddenLength);
    textEmbeds.set(pooledG, pooledLength);
  }
  log("Classifier-free conditioning ready · encoder_hidden_states [2,77,2048] · text_embeds [2,1280] · " + branchDescription + ".");
  return { encoderHidden, textEmbeds };
}

function createRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function createNormalNoise(length, seed, output = new Float32Array(length)) {
  if (!(output instanceof Float32Array) || output.length !== length) {
    throw new Error("Reusable Gaussian-noise buffer has an unexpected size or type.");
  }
  const random = createRng(seed);
  for (let i = 0; i < length; i += 2) {
    const u1 = Math.max(random(), 1e-9);
    const u2 = random();
    const radius = Math.sqrt(-2 * Math.log(u1));
    output[i] = radius * Math.cos(2 * Math.PI * u2);
    if (i + 1 < length) output[i + 1] = radius * Math.sin(2 * Math.PI * u2);
  }
  return output;
}

function createEulerAncestralScheduler(config, steps) {
  const trainSteps = config.num_train_timesteps;
  if (!Number.isInteger(trainSteps) || trainSteps < 2) throw new Error("Invalid scheduler training-step count.");
  if (config.beta_schedule !== "scaled_linear" || config.prediction_type !== "epsilon" ||
      config.timestep_spacing !== "leading" || config.rescale_betas_zero_snr ||
      config.use_karras_sigmas || (config.final_sigmas_type && config.final_sigmas_type !== "zero")) {
    throw new Error("The pinned scheduler config no longer matches the supported Euler ancestral implementation.");
  }

  const alphasCumprod = new Float32Array(trainSteps);
  const start = Math.sqrt(config.beta_start);
  const end = Math.sqrt(config.beta_end);
  let cumulative = 1;
  for (let i = 0; i < trainSteps; i++) {
    const rootBeta = Math.fround(start + (end - start) * (i / (trainSteps - 1)));
    const beta = Math.fround(rootBeta * rootBeta);
    cumulative = Math.fround(cumulative * Math.fround(1 - beta));
    alphasCumprod[i] = cumulative;
  }

  const trainingSigmas = new Float32Array(trainSteps);
  for (let i = 0; i < trainSteps; i++) {
    trainingSigmas[i] = Math.fround(Math.sqrt((1 - alphasCumprod[i]) / alphasCumprod[i]));
  }

  const ratio = Math.floor(trainSteps / steps);
  const offset = Number(config.steps_offset || 0);
  const timesteps = new Float32Array(steps);
  const sigmas = new Float32Array(steps + 1);
  for (let i = 0; i < steps; i++) {
    const timestep = (steps - 1 - i) * ratio + offset;
    if (timestep < 0 || timestep >= trainSteps) throw new Error("Scheduler generated an out-of-range timestep.");
    timesteps[i] = timestep;
    sigmas[i] = trainingSigmas[timestep];
  }
  sigmas[steps] = 0;

  return {
    timesteps,
    sigmas,
    initNoiseSigma: Math.sqrt(sigmas[0] * sigmas[0] + 1),
    scaleModelInput(sample, index) {
      const scale = 1 / Math.sqrt(sigmas[index] * sigmas[index] + 1);
      const output = new Float32Array(sample.length);
      for (let i = 0; i < sample.length; i++) output[i] = sample[i] * scale;
      return output;
    },
    stepParameters(index, cfg, batch) {
      const sigma = sigmas[index];
      const sigmaTo = sigmas[index + 1];
      const sigmaUp = Math.sqrt(Math.max(0, sigmaTo * sigmaTo *
        (sigma * sigma - sigmaTo * sigmaTo) / (sigma * sigma)));
      const sigmaDown = Math.sqrt(Math.max(0, sigmaTo * sigmaTo - sigmaUp * sigmaUp));
      return {
        cfg,
        batch,
        dt: sigmaDown - sigma,
        sigmaUp,
        nextScale: 1 / Math.sqrt(sigmaTo * sigmaTo + 1)
      };
    },
    step(modelOutput, index, sample, noise) {
      const sigma = sigmas[index];
      const sigmaTo = sigmas[index + 1];
      const sigmaUp = Math.sqrt(Math.max(0, sigmaTo * sigmaTo *
        (sigma * sigma - sigmaTo * sigmaTo) / (sigma * sigma)));
      const sigmaDown = Math.sqrt(Math.max(0, sigmaTo * sigmaTo - sigmaUp * sigmaUp));
      const output = new Float32Array(sample.length);
      const dt = sigmaDown - sigma;
      for (let i = 0; i < sample.length; i++) {
        output[i] = sample[i] + modelOutput[i] * dt + noise[i] * sigmaUp;
      }
      return output;
    }
  };
}

function makeBatchSample(scaled, latentHeight, latentWidth, imageCount = 1, branchCount = 2) {
  const oneImageSize = 4 * latentHeight * latentWidth;
  const totalSize = oneImageSize * imageCount;
  if (!scaled || scaled.length !== totalSize || (branchCount !== 1 && branchCount !== 2)) {
    throw new Error("Latent batch dimensions do not match the UNet image/CFG batch.");
  }
  const batch = new Float16Array(totalSize * branchCount);
  batch.set(scaled, 0);
  if (branchCount === 2) batch.set(scaled, totalSize);
  return batch;
}

function makeConditioningBatch(conditioning, imageCount, branchCount, conditioningIndex) {
  const hiddenLength = 77 * 2048;
  const pooledLength = 1280;
  const hidden = new Float16Array(hiddenLength * imageCount * branchCount);
  const pooled = new Float16Array(pooledLength * imageCount * branchCount);
  for (let branch = 0; branch < branchCount; branch++) {
    const sourceBranch = branchCount === 1 ? conditioningIndex : branch;
    const hiddenSource = conditioning.encoderHidden.subarray(
      sourceBranch * hiddenLength, (sourceBranch + 1) * hiddenLength
    );
    const pooledSource = conditioning.textEmbeds.subarray(
      sourceBranch * pooledLength, (sourceBranch + 1) * pooledLength
    );
    for (let image = 0; image < imageCount; image++) {
      const targetIndex = branch * imageCount + image;
      hidden.set(hiddenSource, targetIndex * hiddenLength);
      pooled.set(pooledSource, targetIndex * pooledLength);
    }
  }
  return { hidden, pooled };
}

function makeBatchTimeIds(height, width, batchSize = 2) {
  const one = [height, width, 0, 0, height, width];
  const batch = new Float32Array(batchSize * 6);
  for (let index = 0; index < batchSize; index++) batch.set(one, index * 6);
  return batch;
}

function validateSessionInputs(session, required, label) {
  const missing = required.filter(name => !session.inputNames.includes(name));
  if (missing.length) {
    throw new Error(label + " is missing graph inputs: " + missing.join(", ") +
      ". Actual inputs: " + session.inputNames.join(", "));
  }
}

function validateStaticShape(metadata, expected, label) {
  const declared = metadata?.shape;
  if (!Array.isArray(declared) || declared.length !== expected.length) return;
  const mismatches = [];
  for (let i = 0; i < expected.length; i++) {
    if (typeof declared[i] === "number" && declared[i] > 0 && declared[i] !== expected[i]) {
      mismatches.push("axis " + i + ": graph declares " + declared[i] + ", requested " + expected[i]);
    }
  }
  if (mismatches.length) {
    throw new Error(label + " has a static shape incompatible with the selected size: " + mismatches.join("; ") + ".");
  }
}

function validateGpuTensor(tensor, label, expectedShape) {
  if (!tensor || tensor.type !== "float16" || tensor.location !== "gpu-buffer") {
    throw new Error(label + " was not returned as a GPU-resident FP16 tensor (location: " +
      (tensor?.location || "missing") + "). The ONNX Runtime WebGPU handoff is unavailable.");
  }
  if (!Array.isArray(tensor.dims) || tensor.dims.length !== expectedShape.length ||
      tensor.dims.some((dimension, index) => dimension !== expectedShape[index])) {
    throw new Error(label + " returned shape [" + (tensor?.dims || []).join(", ") +
      "], expected [" + expectedShape.join(", ") + "].");
  }
  if (!tensor.gpuBuffer) throw new Error(label + " reports GPU residency but exposes no GPUBuffer.");
  return tensor;
}

function createGpuInput(device, dataType, data, dims, label, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
  const buffer = device.createBuffer({
    size: Math.max(4, data.byteLength),
    usage
  });
  try {
    device.queue.writeBuffer(buffer, 0, data);
    const tensor = ort.Tensor.fromGpuBuffer(buffer, { dataType, dims });
    return { buffer, tensor, data, label };
  } catch (error) {
    buffer.destroy();
    throw new Error("Could not create GPU input " + label + ": " + (error?.message || error));
  }
}

function createGpuOutputSet(device, shapes, label) {
  const fetches = {};
  const buffers = [];
  try {
    for (const [name, dims] of Object.entries(shapes)) {
      const elements = dims.reduce((product, dimension) => product * dimension, 1);
      const buffer = device.createBuffer({
        size: Math.max(4, elements * 2),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      });
      try {
        fetches[name] = ort.Tensor.fromGpuBuffer(buffer, { dataType: "float16", dims });
        buffers.push(buffer);
      } catch (error) {
        buffer.destroy();
        throw error;
      }
    }
  } catch (error) {
    for (const buffer of buffers) buffer.destroy();
    throw new Error("Could not allocate persistent " + label + " GPU outputs: " + (error?.message || error));
  }
  return {
    fetches,
    destroy() {
      for (const buffer of buffers) buffer.destroy();
      buffers.length = 0;
    }
  };
}

function createGpuInputState(conditioning, width, height, batchSize, conditioningIndex, imageCount = 1, branchCount = 2) {
  const device = ort.env.webgpu?.device;
  if (!device || typeof ort.Tensor.fromGpuBuffer !== "function") {
    throw new Error("ONNX Runtime did not expose the GPUDevice/Tensor.fromGpuBuffer API required for reusable GPU inputs.");
  }
  const latentWidth = width / 8;
  const latentHeight = height / 8;
  const latentLength = 4 * latentHeight * latentWidth;
  const conditioningBatch = makeConditioningBatch(conditioning, imageCount, branchCount, conditioningIndex);
  const entries = {};
  try {
    entries.sample = createGpuInput(device, "float16", new Float16Array(latentLength * batchSize),
      [batchSize, 4, latentHeight, latentWidth], "UNET_A sample");
    entries.timestep = createGpuInput(device, "float32", new Float32Array(batchSize),
      [batchSize], "UNET_A timestep");
    entries.encoderHidden = createGpuInput(device, "float16", conditioningBatch.hidden,
      [batchSize, 77, 2048], "shared encoder_hidden_states");
    entries.textEmbeds = createGpuInput(device, "float16", conditioningBatch.pooled,
      [batchSize, 1280], "UNET_A text_embeds");
    entries.timeIds = createGpuInput(device, "float32", makeBatchTimeIds(height, width, batchSize),
      [batchSize, 6], "UNET_A time_ids");
    entries.cOutput = createGpuInput(device, "float16",
      new Float16Array(latentLength * batchSize), [batchSize, 4, latentHeight, latentWidth],
      "UNET_C reusable output", GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  } catch (error) {
    for (const entry of Object.values(entries)) entry.buffer.destroy();
    throw error;
  }
  return {
    device,
    entries,
    batchSize,
    imageCount,
    branchCount,
    writeStep(sample, timestep) {
      if (sample.byteLength !== entries.sample.buffer.size) {
        throw new Error("Latent input size changed while the UNet engine was resident.");
      }
      entries.timestep.data.fill(timestep);
      device.queue.writeBuffer(entries.sample.buffer, 0, sample);
      device.queue.writeBuffer(entries.timestep.buffer, 0, entries.timestep.data);
    },
    writeTimestep(timestep) {
      entries.timestep.data.fill(timestep);
      device.queue.writeBuffer(entries.timestep.buffer, 0, entries.timestep.data);
    },
    updateConditioning(next, nextConditioningIndex = conditioningIndex) {
      const nextBatch = makeConditioningBatch(next, imageCount, branchCount, nextConditioningIndex);
      if (nextBatch.hidden.byteLength !== entries.encoderHidden.buffer.size ||
          nextBatch.pooled.byteLength !== entries.textEmbeds.buffer.size) {
        throw new Error("Prompt conditioning dimensions changed while the UNet engine was resident.");
      }
      device.queue.writeBuffer(entries.encoderHidden.buffer, 0, nextBatch.hidden);
      device.queue.writeBuffer(entries.textEmbeds.buffer, 0, nextBatch.pooled);
    },
    destroy() {
      for (const entry of Object.values(entries)) {
        if (entry?.buffer) entry.buffer.destroy();
      }
    }
  };
}


async function createGpuScheduler(device, gpuInputs, width, height, batchSize, imageCount = 1, branchCount = 2) {
  const latentSize = 4 * (height / 8) * (width / 8);
  const totalLatentSize = latentSize * imageCount;
  let latentBuffer = null;
  let noiseBuffer = null;
  let uniformBuffer = null;
  try {
    latentBuffer = device.createBuffer({
      size: totalLatentSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    noiseBuffer = device.createBuffer({
      size: totalLatentSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    uniformBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
  } catch (error) {
    latentBuffer?.destroy();
    noiseBuffer?.destroy();
    uniformBuffer?.destroy();
    throw error;
  }
  const shader = device.createShaderModule({ code: `
    struct Params {
      count: u32,
      branches: u32,
      cfg: f32,
      dt: f32,
      sigmaUp: f32,
      nextScale: f32,
      latentSize: u32,
      padding: u32
    };
    @group(0) @binding(0) var<storage, read> prediction: array<u32>;
    @group(0) @binding(1) var<storage, read_write> latents: array<f32>;
    @group(0) @binding(2) var<storage, read> noise: array<f32>;
    @group(0) @binding(3) var<storage, read_write> nextSample: array<u32>;
    @group(0) @binding(4) var<uniform> params: Params;

    fn halfAt(index: u32) -> f32 {
      let packed = unpack2x16float(prediction[index / 2u]);
      return select(packed.x, packed.y, (index & 1u) == 1u);
    }

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let pair = gid.x;
      let first = pair * 2u;
      if (first >= params.count) { return; }
      let second = first + 1u;

      var epsilon0 = halfAt(first);
      var epsilon1 = halfAt(second);
      if (params.branches == 2u) {
        let conditional0 = halfAt(params.count + first);
        let conditional1 = halfAt(params.count + second);
        epsilon0 = epsilon0 + params.cfg * (conditional0 - epsilon0);
        epsilon1 = epsilon1 + params.cfg * (conditional1 - epsilon1);
      }

      let latent0 = latents[first] + epsilon0 * params.dt + noise[first] * params.sigmaUp;
      let latent1 = latents[second] + epsilon1 * params.dt + noise[second] * params.sigmaUp;
      latents[first] = latent0;
      latents[second] = latent1;

      let packed = pack2x16float(vec2<f32>(latent0 * params.nextScale, latent1 * params.nextScale));
      nextSample[pair] = packed;
      if (params.branches == 2u) {
        nextSample[params.count / 2u + pair] = packed;
      }
    }
  ` });
  try {
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: shader, entryPoint: "main" }
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuInputs.entries.cOutput.buffer } },
        { binding: 1, resource: { buffer: latentBuffer } },
        { binding: 2, resource: { buffer: noiseBuffer } },
        { binding: 3, resource: { buffer: gpuInputs.entries.sample.buffer } },
        { binding: 4, resource: { buffer: uniformBuffer } }
      ]
    });
    const paramsData = new ArrayBuffer(32);
    const paramsView = new DataView(paramsData);
    return {
      latentBuffer, noiseBuffer, uniformBuffer, pipeline, bindGroup,
      latentSize, totalLatentSize, batchSize, imageCount, branchCount,
      initialize(latents, scaledLatents) {
        if (latents.length !== totalLatentSize || scaledLatents.length !== totalLatentSize) {
          throw new Error("Initial GPU scheduler latent dimensions do not match the selected image batch.");
        }
        device.queue.writeBuffer(latentBuffer, 0, latents);
        const batchSample = makeBatchSample(
          scaledLatents, height / 8, width / 8, imageCount, branchCount
        );
        device.queue.writeBuffer(gpuInputs.entries.sample.buffer, 0, batchSample);
      },
      step(noise, stepParams) {
        if (noise.length !== totalLatentSize) throw new Error("GPU scheduler noise dimensions changed.");
        device.queue.writeBuffer(noiseBuffer, 0, noise);
        paramsView.setUint32(0, totalLatentSize, true);
        paramsView.setUint32(4, branchCount, true);
        paramsView.setFloat32(8, stepParams.cfg, true);
        paramsView.setFloat32(12, stepParams.dt, true);
        paramsView.setFloat32(16, stepParams.sigmaUp, true);
        paramsView.setFloat32(20, stepParams.nextScale, true);
        paramsView.setUint32(24, totalLatentSize, true);
        paramsView.setUint32(28, 0, true);
        device.queue.writeBuffer(uniformBuffer, 0, paramsData);
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(totalLatentSize / 2 / 256));
        pass.end();
        device.queue.submit([encoder.finish()]);
      },
      async readLatents() {
        const readback = device.createBuffer({
          size: totalLatentSize * 4,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        try {
          const encoder = device.createCommandEncoder();
          encoder.copyBufferToBuffer(latentBuffer, 0, readback, 0, totalLatentSize * 4);
          device.queue.submit([encoder.finish()]);
          await readback.mapAsync(GPUMapMode.READ);
          const result = new Float32Array(readback.getMappedRange().slice(0));
          readback.unmap();
          return result;
        } finally {
          readback.destroy();
        }
      },
      destroy() {
        latentBuffer.destroy();
        noiseBuffer.destroy();
        uniformBuffer.destroy();
      }
    };
  } catch (error) {
    latentBuffer.destroy();
    noiseBuffer.destroy();
    uniformBuffer.destroy();
    throw error;
  }
}

function disposeOutput(outputs, name) {
  const tensor = outputs?.[name];
  if (tensor && typeof tensor.dispose === "function") tensor.dispose();
  if (outputs) delete outputs[name];
}

async function runUnetStep(sessions, conditioning, scaledLatents, timestep, width, height, cfg, noiseStep) {
  const latentWidth = width / 8;
  const latentHeight = height / 8;
  const batch = sessions.batchSize || 2;
  const imageCount = sessions.imageCount || 1;
  const branchCount = sessions.branchCount || (batch > 1 ? 2 : 1);
  const batchSample = sessions.gpuScheduler
    ? null
    : makeBatchSample(scaledLatents, latentHeight, latentWidth, imageCount, branchCount);
  const shape = (channels, h, w) => [batch, channels, h, w];
  const timings = {};
  const gpu = sessions.gpuInputs;
  if (!gpu) throw new Error("Reusable GPU input buffers are missing from the resident UNet engine.");
  let outputsA = null;
  let outputsB = null;
  let outputsC = null;
  const disposeRemaining = outputs => {
    if (!outputs) return;
    for (const tensor of Object.values(outputs)) {
      if (tensor && typeof tensor.dispose === "function") tensor.dispose();
    }
  };
  const usePersistentOutputs = Boolean(sessions.gpuOutputs?.a && sessions.gpuOutputs?.b);

  try {
    if (sessions.gpuScheduler) gpu.writeTimestep(timestep);
    else gpu.writeStep(batchSample, timestep);
    const startedA = performance.now();
    outputsA = await sessions.a.run({
      sample: gpu.entries.sample.tensor,
      timestep: gpu.entries.timestep.tensor,
      encoder_hidden_states: gpu.entries.encoderHidden.tensor,
      text_embeds: gpu.entries.textEmbeds.tensor,
      time_ids: gpu.entries.timeIds.tensor
    }, usePersistentOutputs ? sessions.gpuOutputs.a.fetches : undefined);
    if (!usePersistentOutputs) {
      for (const name of ["hidden", "skip_6", "skip_7", "skip_8"]) disposeOutput(outputsA, name);
    }
    timings.a = (performance.now() - startedA) / 1000;
    checkCancelled();

    validateGpuTensor(outputsA.hidden, "UNET_A hidden", shape(1280, latentHeight / 4, latentWidth / 4));
    validateGpuTensor(outputsA.emb, "UNET_A emb", [batch, 1280]);
    for (const index of [0, 1, 2]) {
      validateGpuTensor(outputsA["skip_" + index], "UNET_A skip_" + index,
        shape(320, latentHeight, latentWidth));
    }
    validateGpuTensor(outputsA.skip_3, "UNET_A skip_3", shape(320, latentHeight / 2, latentWidth / 2));
    validateGpuTensor(outputsA.skip_4, "UNET_A skip_4", shape(640, latentHeight / 2, latentWidth / 2));
    validateGpuTensor(outputsA.skip_5, "UNET_A skip_5", shape(640, latentHeight / 2, latentWidth / 2));
    validateGpuTensor(outputsA.skip_6, "UNET_A skip_6", shape(640, latentHeight / 4, latentWidth / 4));
    validateGpuTensor(outputsA.skip_7, "UNET_A skip_7", shape(1280, latentHeight / 4, latentWidth / 4));
    validateGpuTensor(outputsA.skip_8, "UNET_A skip_8", shape(1280, latentHeight / 4, latentWidth / 4));

    const startedB = performance.now();
    outputsB = await sessions.b.run({
      hidden: outputsA.hidden,
      emb: outputsA.emb,
      encoder_hidden_states: gpu.entries.encoderHidden.tensor,
      skip_6: outputsA.skip_6,
      skip_7: outputsA.skip_7,
      skip_8: outputsA.skip_8
    }, usePersistentOutputs ? sessions.gpuOutputs.b.fetches : undefined);
    timings.b = (performance.now() - startedB) / 1000;
    checkCancelled();
    validateGpuTensor(outputsB.hidden_out, "UNET_B hidden_out",
      shape(1280, latentHeight / 2, latentWidth / 2));

    const startedC = performance.now();
    const feedsC = {
      hidden: outputsB.hidden_out,
      emb: outputsA.emb,
      encoder_hidden_states: gpu.entries.encoderHidden.tensor,
      skip_0: outputsA.skip_0,
      skip_1: outputsA.skip_1,
      skip_2: outputsA.skip_2,
      skip_3: outputsA.skip_3,
      skip_4: outputsA.skip_4,
      skip_5: outputsA.skip_5
    };
    if (sessions.gpuScheduler) {
      outputsC = await sessions.c.run(feedsC, { out_sample: gpu.entries.cOutput.tensor });
      validateGpuTensor(gpu.entries.cOutput.tensor, "UNET_C out_sample",
        [batch, 4, latentHeight, latentWidth]);
    } else {
      outputsC = await sessions.c.run(feedsC);
    }
    timings.c = (performance.now() - startedC) / 1000;
    checkCancelled();

    let guided = null;
    if (!sessions.gpuScheduler) {
      const noisePrediction = fp16Values(outputsC.out_sample, "UNET_C out_sample",
        [batch, 4, latentHeight, latentWidth]);
      const latentSize = 4 * latentHeight * latentWidth;
      const totalLatentSize = latentSize * imageCount;
      guided = new Float32Array(totalLatentSize);
      if (branchCount === 1) {
        guided.set(noisePrediction.subarray(0, totalLatentSize));
      } else {
        for (let i = 0; i < totalLatentSize; i++) {
          const unconditional = noisePrediction[i];
          const conditional = noisePrediction[totalLatentSize + i];
          guided[i] = unconditional + cfg * (conditional - unconditional);
        }
      }
    }

    log("Step denoiser timing · A " + timings.a.toFixed(2) + "s · B " +
      timings.b.toFixed(2) + "s · C " + timings.c.toFixed(2) +
      "s · CFG " + cfg + " · " + imageCount + " image(s) · model batch " + batch + " · A→B→C GPU-resident · reusable GPU inputs · standard dispatch" +
      (sessions.gpuScheduler ? " · GPU CFG/Euler" : " · CPU CFG/Euler"));
    return { guided, timings };
  } finally {
    if (!sessions.gpuScheduler) disposeRemaining(outputsC);
    if (!usePersistentOutputs) {
      disposeRemaining(outputsB);
      disposeRemaining(outputsA);
    }
  }
}

async function releaseSessions(sessions) {
  const failures = [];
  for (const key of ["c", "b", "a"]) {
    const session = sessions[key];
    if (!session) continue;
    try {
      await session.release();
      sessions[key] = null;
      await yieldFrames();
      log("Released UNET_" + key.toUpperCase() + ".");
    } catch (error) {
      const message = "UNET_" + key.toUpperCase() + ": " + (error?.message || error);
      failures.push(message);
      log("ERROR releasing " + message);
    }
  }
  if (failures.length) {
    throw new Error("One or more UNet sessions could not be released: " + failures.join("; "));
  }
  if (sessions.gpuOutputs) {
    for (const outputSet of Object.values(sessions.gpuOutputs)) {
      outputSet?.destroy();
    }
    sessions.gpuOutputs = null;
  }
  if (sessions.gpuScheduler) {
    sessions.gpuScheduler.destroy();
    sessions.gpuScheduler = null;
  }
  if (sessions.gpuInputs) {
    sessions.gpuInputs.destroy();
    sessions.gpuInputs = null;
  }
}

function renderImage(tensor, width, height) {
  if (!Array.isArray(tensor?.dims) || tensor.dims.length !== 4 ||
      tensor.dims[0] !== 1 || tensor.dims[1] !== 3 ||
      tensor.dims[2] !== height || tensor.dims[3] !== width) {
    throw new Error("VAE output shape [" + (tensor?.dims || []).join(", ") +
      "] does not match expected [1, 3, " + height + ", " + width + "].");
  }
  const values = imageTensorValues(tensor, "VAE decoded image");

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: false });
  if (!context) throw new Error("Could not create a 2D canvas context.");
  const pixels = context.createImageData(width, height);
  const planeSize = width * height;
  for (let pixel = 0; pixel < planeSize; pixel++) {
    const sourceR = values[pixel];
    const sourceG = values[planeSize + pixel];
    const sourceB = values[planeSize * 2 + pixel];
    if (!Number.isFinite(sourceR) || !Number.isFinite(sourceG) || !Number.isFinite(sourceB)) {
      throw new Error("VAE decoded image contains a non-finite channel at pixel " + pixel + ".");
    }
    const r = Math.max(0, Math.min(255, Math.round((sourceR * 0.5 + 0.5) * 255)));
    const g = Math.max(0, Math.min(255, Math.round((sourceG * 0.5 + 0.5) * 255)));
    const b = Math.max(0, Math.min(255, Math.round((sourceB * 0.5 + 0.5) * 255)));
    const offset = pixel * 4;
    pixels.data[offset] = r;
    pixels.data[offset + 1] = g;
    pixels.data[offset + 2] = b;
    pixels.data[offset + 3] = 255;
  }
  context.putImageData(pixels, 0, 0);
  return canvas;
}

const GENERATION_SETTINGS_FORMAT = "dragonfiregames.illustrious-generation-settings";
const GENERATION_SETTINGS_VERSION = 4;
const settingsExportButton = document.querySelector("#exportGenerationSettings");
const settingsImportButton = document.querySelector("#importGenerationSettings");
const settingsImportInput = document.querySelector("#importGenerationSettingsFile");
const settingsStatus = document.querySelector("#generationSettingsStatus");

function buildGenerationSettings() {
  const adapterFile = fields.loraFile.files?.[0] || null;
  return {
    format: GENERATION_SETTINGS_FORMAT,
    version: GENERATION_SETTINGS_VERSION,
    exportedAt: new Date().toISOString(),
    model: {
      id: MODEL_ID,
      sampler: "Euler a (Euler ancestral)",
      schedulerConfigUrl: SCHEDULER_URL,
      vaeScalingFactor: VAE_SCALING_FACTOR
    },
    generation: {
      prompt: fields.prompt.value,
      negativePrompt: fields.negative.value,
      positiveSuffix: fields.suffix.value,
      width: fields.width.value === "" ? null : Number(fields.width.value),
      height: fields.height.value === "" ? null : Number(fields.height.value),
      steps: fields.steps.value === "" ? null : Number(fields.steps.value),
      cfgScale: fields.cfg.value === "" ? null : Number(fields.cfg.value),
      seed: fields.seed.value === "" ? null : Number(fields.seed.value),
      batchCount: fields.batch.value === "" ? null : Number(fields.batch.value),
      parallelImages: Number(fields.parallel?.value || 1),
      lora: {
        enabled: fields.lora.checked,
        strength: fields.loraStrength.value === "" ? null : Number(fields.loraStrength.value),
        file: adapterFile ? {
          name: adapterFile.name,
          sizeBytes: adapterFile.size
        } : null
      }
    }
  };
}

function validateGenerationSettings(settings) {
  if (!settings || typeof settings !== "object" ||
      settings.format !== GENERATION_SETTINGS_FORMAT ||
      (settings.version !== 2 && settings.version !== 3 && settings.version !== GENERATION_SETTINGS_VERSION)) {
    throw new Error("Unsupported settings file. Expected " + GENERATION_SETTINGS_FORMAT +
      " version 2 or " + GENERATION_SETTINGS_VERSION + ".");
  }
  if (!settings.model || typeof settings.model.id !== "string") {
    throw new Error("Settings file is missing its model identifier.");
  }
  const generation = settings.generation;
  if (!generation || typeof generation !== "object") {
    throw new Error("Settings file is missing the generation section.");
  }
  for (const key of ["prompt", "negativePrompt", "positiveSuffix"]) {
    if (typeof generation[key] !== "string") {
      throw new Error("Settings field '" + key + "' must be text.");
    }
  }

  const { width, height, steps, cfgScale, seed, lora } = generation;
  const batchCount = generation.batchCount ?? 1;
  const parallelImages = generation.parallelImages ?? 1;
  if (!Number.isInteger(width) || width < 512 || width > 1024 || width % 64 !== 0) {
    throw new Error("Imported width must be 512–1024 and divisible by 64.");
  }
  if (!Number.isInteger(height) || height < 512 || height > 1024 || height % 64 !== 0) {
    throw new Error("Imported height must be 512–1024 and divisible by 64.");
  }
  if (!Number.isInteger(steps) || steps < 1 || steps > 100) {
    throw new Error("Imported steps must be an integer from 1 to 100.");
  }
  if (!Number.isFinite(cfgScale) || cfgScale < 0 || cfgScale > 20) {
    throw new Error("Imported CFG scale must be between 0 and 20.");
  }
  if (!Number.isSafeInteger(seed) || (seed !== -1 && (seed < 0 || seed > 4294967295))) {
    throw new Error("Imported seed must be -1 (random) or an integer from 0 to 4294967295.");
  }
  if (!Number.isInteger(batchCount) || batchCount < 1 || batchCount > 20) {
    throw new Error("Imported batch count must be an integer from 1 to 20.");
  }
  if (!Number.isInteger(parallelImages) || parallelImages < 1 || parallelImages > 4) {
    throw new Error("Imported images per GPU batch must be an integer from 1 to 4.");
  }
  if (seed !== -1 && seed + batchCount - 1 > 4294967295) {
    throw new Error("Imported batch seed range exceeds the maximum 32-bit seed.");
  }
  if (!lora || typeof lora.enabled !== "boolean" ||
      !Number.isFinite(lora.strength) || lora.strength < 0 || lora.strength > 2) {
    throw new Error("Imported LoRA settings are invalid.");
  }
  if (lora.file !== null && (
      !lora.file || typeof lora.file !== "object" ||
      typeof lora.file.name !== "string" || !lora.file.name ||
      !Number.isSafeInteger(lora.file.sizeBytes) || lora.file.sizeBytes < 0)) {
    throw new Error("Imported adapter file metadata is invalid.");
  }
  return generation;
}

function downloadGenerationSettings() {
  const settings = buildGenerationSettings();
  const blob = new Blob([JSON.stringify(settings, null, 2) + "\n"], {
    type: "application/json"
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "illustrious-settings-" + settings.generation.width + "x" +
    settings.generation.height + "-batch-" + (settings.generation.batchCount ?? 1) +
    "-seed-" + (settings.generation.seed === -1 ? "random" : settings.generation.seed) + ".json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  settingsStatus.textContent = "Exported generation settings JSON. The LoRA .safetensors file is referenced by name and size, not embedded.";
}

async function importGenerationSettings(file) {
  const parsed = JSON.parse(await file.text());
  const generation = validateGenerationSettings(parsed);
  const adapterFile = generation.lora.file;

  fields.prompt.value = generation.prompt;
  fields.negative.value = generation.negativePrompt;
  fields.suffix.value = generation.positiveSuffix;
  fields.width.value = String(generation.width);
  fields.height.value = String(generation.height);
  fields.steps.value = String(generation.steps);
  fields.cfg.value = String(generation.cfgScale);
  fields.seed.value = String(generation.seed);
  fields.batch.value = String(generation.batchCount ?? 1);
  if (fields.parallel) fields.parallel.value = String(generation.parallelImages ?? 1);
  fields.lora.checked = generation.lora.enabled;
  fields.loraStrength.value = String(generation.lora.strength);
  fields.loraStrength.disabled = !generation.lora.enabled;

  expectedAdapterFile = adapterFile ? {
    name: adapterFile.name,
    sizeBytes: adapterFile.sizeBytes
  } : null;
  const currentAdapterFile = fields.loraFile.files?.[0] || null;
  if (expectedAdapterFile && !adapterFileMatches(currentAdapterFile, expectedAdapterFile)) {
    fields.loraFile.value = "";
  }
  updateAdapterFileLabel();

  const notes = [];
  if (parsed.model.id !== MODEL_ID) {
    notes.push("Model mismatch: this file targets " + parsed.model.id + ", while this page uses " + MODEL_ID + ".");
  }
  if (expectedAdapterFile && !adapterFileMatches(fields.loraFile.files?.[0] || null, expectedAdapterFile)) {
    notes.push("Re-select the LoRA file " + expectedAdapterFile.name + " (" +
      formatBytes(expectedAdapterFile.sizeBytes) + "); local adapter files are not stored in JSON.");
  } else if (generation.lora.enabled && !expectedAdapterFile) {
    notes.push("LoRA is enabled, but no adapter file was recorded; select the matching .safetensors file.");
  }
  settingsStatus.textContent = "Imported settings JSON. " +
    (notes.length ? notes.join(" ") : "All generation fields were restored.");
}

function resolveGenerationSeed(seed) {
  if (seed !== -1) return seed;
  if (!globalThis.crypto?.getRandomValues) {
    throw new Error("Secure random seed generation is unavailable in this browser.");
  }
  const values = new Uint32Array(1);
  globalThis.crypto.getRandomValues(values);
  return values[0];
}

function readOptions() {
  const prompt = fields.prompt.value.trim();
  const negative = fields.negative.value.trim();
  const suffix = fields.suffix.value.trim();
  const width = Number(fields.width.value);
  const height = Number(fields.height.value);
  const steps = Number(fields.steps.value);
  const cfg = Number(fields.cfg.value);
  const seed = Number(fields.seed.value);
  const batchCount = Number(fields.batch.value);
  if (!prompt) throw new Error("Enter a prompt first.");
  if (!Number.isInteger(width) || width < 512 || width > 1024 || width % 64 !== 0) {
    throw new Error("Width must be 512–1024 and divisible by 64.");
  }
  if (!Number.isInteger(height) || height < 512 || height > 1024 || height % 64 !== 0) {
    throw new Error("Height must be 512–1024 and divisible by 64.");
  }
  if (!Number.isInteger(steps) || steps < 1 || steps > 100) throw new Error("Steps must be between 1 and 100.");
  if (!Number.isFinite(cfg) || cfg < 0 || cfg > 20) throw new Error("CFG scale must be between 0 and 20.");
  if (!Number.isSafeInteger(seed) || (seed !== -1 && (seed < 0 || seed > 4294967295))) {
    throw new Error("Seed must be -1 (random) or an integer from 0 to 4294967295.");
  }
  if (!Number.isInteger(batchCount) || batchCount < 1 || batchCount > 20) {
    throw new Error("Batch must be an integer from 1 to 20.");
  }
  if (seed !== -1 && seed + batchCount - 1 > 4294967295) {
    throw new Error("Batch seed range exceeds the maximum 32-bit seed. Lower the batch count or choose a smaller starting seed.");
  }
  if (fields.lora.checked) {
    const adapterFile = fields.loraFile.files?.[0] || null;
    if (!adapterFile) throw new Error("Choose a LoRA .safetensors file in the adapter selector above.");
    if (expectedAdapterFile && !adapterFileMatches(adapterFile, expectedAdapterFile)) {
      throw new Error("The selected LoRA file does not match the adapter recorded in the imported settings JSON. Select " +
        expectedAdapterFile.name + " (" + formatBytes(expectedAdapterFile.sizeBytes) + ").");
    }
    const strength = Number(fields.loraStrength.value);
    if (!Number.isFinite(strength) || strength < 0 || strength > 2) {
      throw new Error("LoRA strength must be between 0 and 2.");
    }
  }
  const parallelCount = Number(fields.parallel?.value || 1);
  if (!Number.isInteger(parallelCount) || parallelCount < 1 || parallelCount > 4) {
    throw new Error("Images per GPU batch must be an integer from 1 to 4.");
  }
  return {
    prompt: prompt + (suffix ? ", " + suffix : ""),
    negative,
    width,
    height,
    steps,
    cfg,
    seed,
    batchCount,
    parallelCount,
    lora: fields.lora.checked ? {
      file: fields.loraFile.files?.[0] || null,
      strength: Number(fields.loraStrength.value)
    } : null
  };
}


function adapterCacheKey(lora) {
  if (!lora) return "base";
  const file = lora.file;
  return [
    file.name,
    file.size,
    file.lastModified,
    lora.strength
  ].join("|");
}

function conditioningCacheKey(options) {
  return JSON.stringify([
    options.prompt,
    options.negative,
    options.cfg === 0 ? 0 : options.cfg === 1 ? 1 : 2,
    adapterCacheKey(options.lora)
  ]);
}

async function getConditioning(manifest, options) {
  const key = conditioningCacheKey(options);
  if (conditioningCache.has(key)) {
    const cached = conditioningCache.get(key);
    conditioningCache.delete(key);
    conditioningCache.set(key, cached);
    log("Reused cached CLIP conditioning · positive/negative prompt and adapter match.");
    return cached;
  }
  const conditioning = await encodePrompts(
    manifest,
    options.prompt,
    options.negative,
    options.lora,
    options.cfg
  );
  conditioningCache.set(key, conditioning);
  while (conditioningCache.size > 8) {
    conditioningCache.delete(conditioningCache.keys().next().value);
  }
  log("Cached prompt conditioning · " + conditioningCache.size + "/8 entries.");
  return conditioning;
}

async function buildUnetSessions(
  manifest,
  lora,
  conditioning,
  width,
  height,
  batchSize,
  imageCount,
  branchCount,
  conditioningIndex,
  patchCache,
  useGpuScheduler = true,
  enableGraphCapture = false,
  preferredLayout = null,
  validationMode = null
) {
  const sessions = {
    a: null, b: null, c: null, gpuInputs: null, gpuOutputs: { a: null, b: null },
    batchSize, imageCount, branchCount
  };
  const patchedComponent = async name => {
    if (!lora) return null;
    if (!patchCache.has(name)) {
      patchCache.set(name, await prepareLoraComponent(
        lora.file, manifest, name, lora.strength, log, checkCancelled
      ));
    }
    return patchCache.get(name);
  };
  const outputsA = {
    hidden: "gpu-buffer",
    emb: "gpu-buffer",
    skip_0: "gpu-buffer",
    skip_1: "gpu-buffer",
    skip_2: "gpu-buffer",
    skip_3: "gpu-buffer",
    skip_4: "gpu-buffer",
    skip_5: "gpu-buffer",
    skip_6: "gpu-buffer",
    skip_7: "gpu-buffer",
    skip_8: "gpu-buffer"
  };
  const outputsB = { hidden_out: "gpu-buffer" };
  try {
    const patchedA = await patchedComponent("unet_a");
    sessions.a = await loadSession(
      manifest, "unet_a", "UNET_A", patchedA?.externalData, outputsA,
      enableGraphCapture, preferredLayout, validationMode
    );
    sessions.gpuInputs = createGpuInputState(
      conditioning, width, height, batchSize, conditioningIndex, imageCount, branchCount
    );
    checkCancelled();

    const patchedB = await patchedComponent("unet_b");
    sessions.b = await loadSession(
      manifest, "unet_b", "UNET_B", patchedB?.externalData, outputsB,
      enableGraphCapture, preferredLayout, validationMode
    );
    checkCancelled();

    const patchedC = await patchedComponent("unet_c");
    if (useGpuScheduler) {
      try {
        sessions.gpuScheduler = await createGpuScheduler(
          ort.env.webgpu.device, sessions.gpuInputs, width, height, batchSize, imageCount, branchCount
        );
        log("GPU-resident Euler scheduler initialized · CFG and latent update stay on WebGPU.");
      } catch (error) {
        sessions.gpuScheduler = null;
        sessions.gpuInputs.entries.cOutput.buffer.destroy();
        sessions.gpuInputs.entries.cOutput = null;
        log("GPU scheduler unavailable; retaining the CPU scheduler. Reason: " +
          (error?.message || String(error)));
      }
    } else {
      sessions.gpuInputs.entries.cOutput.buffer.destroy();
      sessions.gpuInputs.entries.cOutput = null;
      log("Masked infill selected · CPU latent scheduler enabled for per-step source blending.");
    }
    sessions.c = await loadSession(
      manifest, "unet_c", "UNET_C", patchedC?.externalData,
      sessions.gpuScheduler ? "gpu-buffer" : null,
      enableGraphCapture,
      preferredLayout,
      validationMode
    );

    validateSessionInputs(sessions.a, ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"], "UNET_A");
    validateSessionInputs(sessions.b, ["hidden", "emb", "encoder_hidden_states", "skip_6", "skip_7", "skip_8"], "UNET_B");
    validateSessionInputs(sessions.c, ["hidden", "emb", "encoder_hidden_states", "skip_0", "skip_1", "skip_2", "skip_3", "skip_4", "skip_5"], "UNET_C");

    const latentWidth = width / 8;
    const latentHeight = height / 8;
    try {
      sessions.gpuOutputs.a = createGpuOutputSet(ort.env.webgpu.device, {
        hidden: [batchSize, 1280, latentHeight / 4, latentWidth / 4],
        emb: [batchSize, 1280],
        skip_0: [batchSize, 320, latentHeight, latentWidth],
        skip_1: [batchSize, 320, latentHeight, latentWidth],
        skip_2: [batchSize, 320, latentHeight, latentWidth],
        skip_3: [batchSize, 320, latentHeight / 2, latentWidth / 2],
        skip_4: [batchSize, 640, latentHeight / 2, latentWidth / 2],
        skip_5: [batchSize, 640, latentHeight / 2, latentWidth / 2],
        skip_6: [batchSize, 640, latentHeight / 4, latentWidth / 4],
        skip_7: [batchSize, 1280, latentHeight / 4, latentWidth / 4],
        skip_8: [batchSize, 1280, latentHeight / 4, latentWidth / 4]
      }, "UNET_A");
      sessions.gpuOutputs.b = createGpuOutputSet(ort.env.webgpu.device, {
        hidden_out: [batchSize, 1280, latentHeight / 2, latentWidth / 2]
      }, "UNET_B");
      for (const name of Object.keys(sessions.gpuOutputs.a.fetches)) {
        if (!sessions.a.outputNames.includes(name)) throw new Error("UNET_A is missing expected output " + name + ".");
      }
      for (const name of Object.keys(sessions.gpuOutputs.b.fetches)) {
        if (!sessions.b.outputNames.includes(name)) throw new Error("UNET_B is missing expected output " + name + ".");
      }
      log("Persistent UNET_A/B output buffers allocated for reuse across denoising steps.");
    } catch (error) {
      sessions.gpuOutputs.a?.destroy();
      sessions.gpuOutputs.b?.destroy();
      sessions.gpuOutputs = { a: null, b: null };
      log("Persistent UNet output buffers unavailable; using runtime-managed outputs. Reason: " +
        (error?.message || String(error)));
    }
    return sessions;
  } catch (error) {
    await releaseSessions(sessions);
    throw error;
  }
}

async function ensureUnetSessions(
  manifest, lora, conditioning, width, height, cfg, useGpuScheduler = true,
  imageCount = 1, validationMode = preferredUnetValidationMode
) {
  const branchCount = cfg === 0 || cfg === 1 ? 1 : 2;
  const batchSize = branchCount * imageCount;
  const conditioningIndex = cfg === 1 ? 1 : 0;
  const key = adapterCacheKey(lora) + "|" + width + "x" + height + "|images" + imageCount +
    "|branches" + branchCount + "|batch" + batchSize +
    (useGpuScheduler ? "|gpu-scheduler" : "|cpu-scheduler") +
    (graphCaptureEnabled ? "|graph-capture" : "|standard-dispatch") +
    "|" + (preferredUnetLayout || "default-layout") +
    "|validation:" + (validationMode || "ort-default");
  if (residentUnetSessions && residentUnetKey === key) {
    log("Reusing resident UNET_A, UNET_B, and UNET_C sessions.");
    residentUnetSessions.gpuInputs.updateConditioning(conditioning, conditioningIndex);
    if (!residentEngineLock) {
      releaseHeavyModelTest("full Illustrious image generation");
      acquireHeavyModelTest("resident Illustrious inference engine");
      residentEngineLock = true;
    }
    return residentUnetSessions;
  }

  if (residentUnetSessions) {
    log("Adapter or resolution changed · releasing the previous resident UNet before rebuilding it.");
    await releaseSessions(residentUnetSessions);
    residentUnetSessions = null;
    residentUnetKey = null;
  }
  if (residentVaeSession) {
    await releaseVaeSession("UNet sessions are being rebuilt");
  }

  setProgress(8, "Preparing UNet partitions");
  setStatus("Preparing UNET_A, UNET_B, and UNET_C. This requires several GiB of memory…");
  const patchCache = new Map();
  let sessions;
  try {
    const runWarmup = async () => {
      const warmupStarted = performance.now();
      const warmupLatents = new Float32Array(4 * (height / 8) * (width / 8) * imageCount);
      log("Compiling UNet GPU pipelines with one disposable zero-latent warm-up" +
        (graphCaptureEnabled ? " and recording the static WebGPU command graphs…" : "…"));
      const warmup = await runUnetStep(
        sessions, conditioning, warmupLatents, 0, width, height, cfg, 0
      );
      log("UNet pipeline warm-up finished in " + ((performance.now() - warmupStarted) / 1000).toFixed(2) +
        "s · A " + warmup.timings.a.toFixed(2) + "s · B " + warmup.timings.b.toFixed(2) +
        "s · C " + warmup.timings.c.toFixed(2) + "s. Subsequent images reuse the compiled pipelines.");
    };

    try {
      sessions = await buildUnetSessions(
        manifest, lora, conditioning, width, height, batchSize, imageCount, branchCount, conditioningIndex,
        patchCache, useGpuScheduler, graphCaptureEnabled, preferredUnetLayout,
        validationMode
      );
      log("UNET_A/B/C ready · " + (graphCaptureEnabled ? "graph-captured" : "standard") +
        " WebGPU dispatch with persistent GPU inputs and GPU-resident partition handoffs." +
        (branchCount === 1 ? " Single-branch CFG path." : " Classifier-free guidance path.") +
        " · " + imageCount + " image(s) · model batch=" + batchSize + ".");
      await runWarmup();
    } catch (captureError) {
      if (!graphCaptureEnabled) throw captureError;
      if (sessions) {
        await releaseSessions(sessions);
        sessions = null;
      }
      log("Graph capture was unavailable for this UNet shape/operator combination; retrying with standard WebGPU dispatch. Reason: " +
        (captureError?.message || String(captureError)));
      sessions = await buildUnetSessions(
        manifest, lora, conditioning, width, height, batchSize, imageCount, branchCount, conditioningIndex,
        patchCache, useGpuScheduler, false, preferredUnetLayout,
        validationMode
      );
      await runWarmup();
    }

    if (lora) {
      releaseLoraPayload(lora.file);
      log("Released the temporary in-memory LoRA factor copy after all five components were processed.");
    }
    residentUnetSessions = sessions;
    residentUnetKey = key;
    releaseEngineButton.disabled = false;
    log("UNet engine prepared · all three sessions remain resident; one zero-latent pipeline-compilation pass completed.");

    if (!residentEngineLock) {
      releaseHeavyModelTest("full Illustrious image generation");
      acquireHeavyModelTest("resident Illustrious inference engine");
      residentEngineLock = true;
    }
    return residentUnetSessions;
  } catch (error) {
    if (sessions) await releaseSessions(sessions);
    throw error;
  }
}

async function ensureVaeSession(manifest) {
  if (residentVaeSession) {
    log("Reusing resident VAE decoder session.");
    return residentVaeSession;
  }
  residentVaeSession = await loadSession(manifest, "vae_decoder", "VAE decoder");
  return residentVaeSession;
}

async function releaseVaeSession(reason) {
  if (!residentVaeSession) return;
  const session = residentVaeSession;
  log("Releasing resident VAE decoder" + (reason ? " · " + reason : "") + " to free GPU memory.");
  await session.release();
  residentVaeSession = null;
  await yieldFrames();
  log("Released VAE decoder session.");
}

async function releaseInferenceEngine() {
  const failures = [];
  if (residentVaeSession) {
    try {
      await residentVaeSession.release();
      residentVaeSession = null;
      await yieldFrames();
      log("Released resident VAE decoder.");
    } catch (error) {
      failures.push("VAE decoder: " + (error?.message || error));
      log("ERROR releasing VAE decoder: " + (error?.message || error));
    }
  }
  if (residentUnetSessions) {
    try {
      await releaseSessions(residentUnetSessions);
      residentUnetSessions = null;
      residentUnetKey = null;
    } catch (error) {
      failures.push(error?.message || String(error));
    }
  }
  if (failures.length) {
    throw new Error("Engine release incomplete; GPU memory may still be occupied. " + failures.join(" · "));
  }
  if (residentEngineLock) {
    releaseHeavyModelTest("resident Illustrious inference engine");
    residentEngineLock = false;
  }
  conditioningCache.clear();
  log("Inference engine released. GPU model sessions are no longer resident.");
}

async function prepareInferenceEngine() {
  generationStarted = performance.now();
  logOutput.textContent = "";
  cancelRequested = false;
  const acquiredHere = !residentEngineLock;
  prepareEngineButton.disabled = true;
  releaseEngineButton.disabled = true;
  button.disabled = true;
  cancelButton.disabled = false;
  try {
    const options = readOptions();
    if (acquiredHere) {
      acquireHeavyModelTest("resident Illustrious inference engine");
      residentEngineLock = true;
    }
    if (typeof Float16Array !== "function") throw new Error("This browser does not provide Float16Array.");
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("The browser did not return a WebGPU adapter.");
    if (ort.env.webgpu && !ort.env.webgpu.device) ort.env.webgpu.adapter = adapter;
    const adapterInfo = adapter.info || {};
    const adapterLabel = [adapterInfo.vendor, adapterInfo.architecture, adapterInfo.device, adapterInfo.description]
      .filter(Boolean).join(" · ");
    if (adapterLabel) log("WebGPU adapter identity · " + adapterLabel);
    log("WebGPU adapter ready · maxBufferSize " + formatBytes(adapter.limits?.maxBufferSize || 0) +
      " · maxStorageBufferBindingSize " + formatBytes(adapter.limits?.maxStorageBufferBindingSize || 0));

    const manifest = await fetchJSON(MANIFEST_URL, "Model manifest");
    log("Using the local model cache for CLIP and UNet resources. Missing files will be reported before any fallback download.");
    setStatus("Precomputing prompt conditioning…");
    setProgress(7, "Precomputing prompt conditioning");
    const conditioning = await getConditioning(manifest, options);
    checkCancelled();

    const infillMode = Boolean(window.__illustriousInfill);
    setStatus("Preparing resident UNet sessions…");
    await ensureUnetSessions(
      manifest, options.lora, conditioning, options.width, options.height, options.cfg,
      !infillMode, Math.min(options.parallelCount, options.batchCount),
      preferredUnetValidationMode
    );
    checkCancelled();

    // Keep the multi-part UNet resident, but do not pin the VAE beside its
    // 4.79 GiB of external weights. Load VAE only for decode, then release it.
    if (keepVaeResident) {
      log("VAE retention enabled · the decoder will remain resident after its first use.");
    } else {
      await releaseVaeSession("UNet engine takes priority");
    }
    setProgress(100, "Engine ready");
    setStatus("Inference engine prepared · UNet sessions are resident; prompt conditioning is cached.", "success");
    log("Ready for repeated generation. The VAE decoder will be loaded only after denoising to reduce GPU memory contention.");
  } catch (error) {
    console.error("[Illustrious] Engine preparation failed:", error);
    setStatus("Engine preparation failed: " + (error?.message || String(error)), "error");
    log("ERROR: " + (error?.stack || error?.message || String(error)));
    if (acquiredHere && !residentUnetSessions && !residentVaeSession) {
      releaseHeavyModelTest("resident Illustrious inference engine");
      residentEngineLock = false;
    }
  } finally {
    prepareEngineButton.disabled = false;
    releaseEngineButton.disabled = !residentEngineLock;
    button.disabled = false;
    cancelButton.disabled = true;
  }
}


function buildInfillLatentMask(maskCanvas, latentWidth, latentHeight) {
  if (!maskCanvas || maskCanvas.width < 1 || maskCanvas.height < 1) {
    throw new Error("The brush mask is missing.");
  }
  const small = document.createElement("canvas");
  small.width = latentWidth;
  small.height = latentHeight;
  const context = small.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Could not create a latent mask canvas.");
  context.drawImage(maskCanvas, 0, 0, latentWidth, latentHeight);
  const pixels = context.getImageData(0, 0, latentWidth, latentHeight).data;
  const mask = new Float32Array(latentWidth * latentHeight);
  let painted = 0;
  for (let i = 0; i < mask.length; i++) {
    mask[i] = pixels[i * 4 + 3] / 255;
    if (mask[i] > 0.01) painted++;
  }
  if (!painted) throw new Error("Paint the area to replace before starting infill.");
  return mask;
}

async function encodeInfillSource(manifest, sourceCanvas, width, height) {
  const component = manifest?.components?.vae_encoder;
  if (!component?.onnx || !Array.isArray(component.shards) || !component.shards.length) {
    throw new Error("The Illustrious model manifest does not provide a VAE encoder.");
  }
  if (sourceCanvas.width !== width || sourceCanvas.height !== height) {
    throw new Error("The source image canvas does not match the selected generation resolution.");
  }

  const started = performance.now();
  const session = await loadSession(manifest, "vae_encoder", "VAE encoder");
  let outputs = null;
  try {
    if (session.inputNames.length !== 1) {
      throw new Error("Expected one VAE encoder input, found: " + session.inputNames.join(", ") + ".");
    }
    const inputName = session.inputNames[0];
    const inputIndex = session.inputNames.indexOf(inputName);
    validateStaticShape(session.inputMetadata?.[inputIndex], [1, 3, height, width], "VAE encoder input");
    const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
    const pixels = sourceContext.getImageData(0, 0, width, height).data;
    const planeSize = width * height;
    const input = new Float16Array(3 * planeSize);
    for (let i = 0; i < planeSize; i++) {
      input[i] = pixels[i * 4] / 127.5 - 1;
      input[planeSize + i] = pixels[i * 4 + 1] / 127.5 - 1;
      input[planeSize * 2 + i] = pixels[i * 4 + 2] / 127.5 - 1;
    }

    log("Encoding source image " + width + "×" + height + " into SDXL latents…");
    outputs = await session.run({
      [inputName]: new ort.Tensor("float16", input, [1, 3, height, width])
    });
    const expectedWidth = width / 8;
    const expectedHeight = height / 8;
    const outputName = session.outputNames.find(name => {
      const dims = outputs[name]?.dims;
      return Array.isArray(dims) && dims.length === 4 &&
        dims[0] === 1 && (dims[1] === 4 || dims[1] === 8) &&
        dims[2] === expectedHeight && dims[3] === expectedWidth;
    });
    if (!outputName) {
      const shapes = session.outputNames.map(name => name + " [" +
        (outputs[name]?.dims || []).join(", ") + "]").join("; ");
      throw new Error("VAE encoder did not return a [1,4|8," + expectedHeight + "," +
        expectedWidth + "] latent tensor. Outputs: " + shapes);
    }
    const tensor = outputs[outputName];
    const values = imageTensorValues(tensor, "VAE encoder output " + outputName);
    const channels = tensor.dims[1];
    const latentSize = 4 * expectedHeight * expectedWidth;
    const latents = new Float32Array(latentSize);
    // Eight-channel encoder outputs contain mean/log-variance moments. Use the
    // deterministic mean half; four-channel outputs are already latent samples.
    for (let i = 0; i < latentSize; i++) {
      latents[i] = Number(values[i]) * VAE_SCALING_FACTOR;
      if (!Number.isFinite(latents[i])) throw new Error("VAE encoder returned a non-finite latent at " + i + ".");
    }
    log("Source image encoded · " + channels + " output channels · " +
      ((performance.now() - started) / 1000).toFixed(2) + "s · latent scaling " + VAE_SCALING_FACTOR + ".");
    return latents;
  } finally {
    if (outputs) {
      for (const tensor of Object.values(outputs)) {
        if (tensor && typeof tensor.dispose === "function") tensor.dispose();
      }
    }
    await session.release();
    await yieldFrames();
    log("Released VAE encoder to free GPU memory before UNet inference.");
  }
}

function blendUnmaskedLatents(latents, sourceLatents, sourceNoise, mask, sigma) {
  const planeSize = mask.length;
  for (let channel = 0; channel < 4; channel++) {
    const offset = channel * planeSize;
    for (let pixel = 0; pixel < planeSize; pixel++) {
      const keep = 1 - mask[pixel];
      if (keep <= 0) continue;
      const index = offset + pixel;
      const preserved = sourceLatents[index] + sourceNoise[index] * sigma;
      latents[index] = latents[index] * mask[pixel] + preserved * keep;
    }
  }
}

function compositeInfillPixels(generatedCanvas, sourceCanvas, maskCanvas) {
  const width = generatedCanvas.width;
  const height = generatedCanvas.height;
  if (sourceCanvas.width !== width || sourceCanvas.height !== height ||
      maskCanvas.width !== width || maskCanvas.height !== height) {
    throw new Error("Infill source, mask, and generated image dimensions do not match.");
  }
  const generatedContext = generatedCanvas.getContext("2d", { willReadFrequently: true });
  const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
  const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true });
  const generated = generatedContext.getImageData(0, 0, width, height);
  const source = sourceContext.getImageData(0, 0, width, height);
  const mask = maskContext.getImageData(0, 0, width, height);
  for (let i = 0; i < generated.data.length; i += 4) {
    const alpha = mask.data[i + 3] / 255;
    if (alpha <= 0) {
      generated.data[i] = source.data[i];
      generated.data[i + 1] = source.data[i + 1];
      generated.data[i + 2] = source.data[i + 2];
    } else if (alpha < 1) {
      const keep = 1 - alpha;
      generated.data[i] = Math.round(generated.data[i] * alpha + source.data[i] * keep);
      generated.data[i + 1] = Math.round(generated.data[i + 1] * alpha + source.data[i + 1] * keep);
      generated.data[i + 2] = Math.round(generated.data[i + 2] * alpha + source.data[i + 2] * keep);
    }
    generated.data[i + 3] = 255;
  }
  generatedContext.putImageData(generated, 0, 0);
  return generatedCanvas;
}

async function generateMany(imageOptionsList, infillData = null, onImage = null) {
  if (!Array.isArray(imageOptionsList) || !imageOptionsList.length) {
    throw new Error("At least one image configuration is required.");
  }
  const options = imageOptionsList[0];
  const imageCount = imageOptionsList.length;
  if (imageCount > 1 && infillData) {
    throw new Error("Parallel generation is not enabled for brush infill.");
  }
  for (const imageOptions of imageOptionsList) {
    if (imageOptions.width !== options.width || imageOptions.height !== options.height ||
        imageOptions.steps !== options.steps || imageOptions.cfg !== options.cfg ||
        imageOptions.prompt !== options.prompt || imageOptions.negative !== options.negative ||
        adapterCacheKey(imageOptions.lora) !== adapterCacheKey(options.lora)) {
      throw new Error("Images in one GPU batch must share prompt, dimensions, steps, CFG, and adapter settings.");
    }
  }

  const manifest = await fetchJSON(MANIFEST_URL, "Model manifest");
  setProgress(1, "Reading cached model metadata");
  const schedulerConfig = await fetchJSON(SCHEDULER_URL, "Scheduler config");
  const latentWidth = options.width / 8;
  const latentHeight = options.height / 8;
  const latentSize = 4 * latentWidth * latentHeight;
  const totalLatentSize = latentSize * imageCount;
  let sourceLatents = null;
  let latentMask = null;
  let sourceNoise = null;
  let startStep = 0;
  if (infillData) {
    if (!infillData.sourceCanvas || !infillData.maskCanvas) {
      throw new Error("Brush infill requires both a source image and a mask canvas.");
    }
    const strength = Number(infillData.strength);
    if (!Number.isFinite(strength) || strength < 0.1 || strength > 1) {
      throw new Error("Infill strength must be between 0.1 and 1.");
    }
    latentMask = buildInfillLatentMask(infillData.maskCanvas, latentWidth, latentHeight);
    startStep = Math.max(0, Math.min(options.steps - 1, Math.floor((1 - strength) * options.steps)));
    log("Brush infill · strength " + strength.toFixed(2) + " · starting at denoising step " +
      (startStep + 1) + "/" + options.steps + " · masked latent blending (not a dedicated inpainting UNet).");
    setProgress(6, "Encoding source image");
    setStatus("Encoding the uploaded image with the SDXL VAE encoder…");
    sourceLatents = await encodeInfillSource(manifest, infillData.sourceCanvas, options.width, options.height);
    sourceNoise = createNormalNoise(latentSize, options.seed ^ 0x5f3759df);
  }

  log(runtimeLabel + ".");
  if (runtimeNote) log(runtimeNote);
  log("UNet preferred layout · " + (preferredUnetLayout || "NCHW (default)") + ".");
  log("UNet validation mode · " + (preferredUnetValidationMode || "ORT default (basic)") +
    (preferredUnetValidationMode === "wgpuOnly" ? " · retains WebGPU/Dawn validation; skips additional ONNX basic validation." : "."));
  if (graphCaptureEnabled) log("Static UNet WebGPU graph capture requested · initialization falls back to standard dispatch if capture fails.");
  if (gpuProfilingRequested) {
    log("Per-operator GPU profiling is not exposed to JavaScript by the bundled ONNX Runtime 1.31.1 native WebGPU provider. No [profiling] entries are expected; the A/B/C wall-clock timings below remain available.");
  }
  log("Model " + (manifest.packageId || MODEL_ID) + " · WebGPU · " +
    options.width + "×" + options.height + " · " + options.steps + " steps.");
  log("Prompt: " + options.prompt);
  log("Negative: " + (options.negative || "(empty)"));
  log("Image seeds · " + imageOptionsList.map(item => item.seed).join(", ") + " · CFG " + options.cfg);
  log("Shared prompt conditioning, resident UNet sessions, and reusable GPU buffers are used across " +
    imageCount + " image(s) in one model batch.");
  if (options.lora) {
    log("LoRA enabled · " + options.lora.file.name + " · strength " + options.lora.strength.toFixed(3) + " · compatible mapped weights will be patched.");
  } else {
    log("LoRA disabled · using unmodified WAI Illustrious v17 base weights.");
  }

  checkCancelled();
  setProgress(2, "Encoding prompts");
  setStatus("Encoding positive and negative prompts…");
  const conditioning = await getConditioning(manifest, options);
  checkCancelled();

  const scheduler = createEulerAncestralScheduler(schedulerConfig, options.steps);
  log("Euler ancestral scheduler · " + schedulerConfig.num_train_timesteps +
    " train steps · timestep " + scheduler.timesteps[0] + " → " +
    scheduler.timesteps[scheduler.timesteps.length - 1] +
    " · initial sigma " + scheduler.initNoiseSigma.toFixed(5) + ".");
  let latents;
  if (infillData) {
    latents = new Float32Array(latentSize);
    const sigma = scheduler.sigmas[startStep];
    for (let i = 0; i < latentSize; i++) latents[i] = sourceLatents[i] + sourceNoise[i] * sigma;
  } else {
    latents = new Float32Array(totalLatentSize);
    for (let image = 0; image < imageCount; image++) {
      const imageNoise = createNormalNoise(latentSize, imageOptionsList[image].seed);
      const offset = image * latentSize;
      for (let i = 0; i < latentSize; i++) {
        latents[offset + i] = imageNoise[i] * scheduler.initNoiseSigma;
      }
    }
  }

  const parts = ["unet_a", "unet_b", "unet_c"];
  const unetBytes = parts.reduce((sum, key) => sum + Number(manifest.components?.[key]?.bytes || 0), 0);
  log("UNet manifest payload total " + formatBytes(unetBytes) +
    " (not a direct VRAM measurement). Loading all partitions together…");
  if (keepVaeResident) {
    if (residentVaeSession) log("Reusing retained VAE decoder alongside the resident UNet.");
  } else {
    await releaseVaeSession("starting UNet denoising");
  }
  const sessions = await ensureUnetSessions(
    manifest, options.lora, conditioning, options.width, options.height, options.cfg,
    !infillData, imageCount,
    preferredUnetValidationMode
  );
  const gpuScheduler = sessions.gpuScheduler;
  if (gpuScheduler) {
    const initialScaled = scheduler.scaleModelInput(latents, startStep);
    gpuScheduler.initialize(latents, new Float16Array(initialScaled));
  } else if (imageCount > 1) {
    throw new Error("Parallel image batches require the GPU-resident Euler scheduler; reduce Images per GPU batch to 1.");
  }
  let totalUnetExecutionSeconds = 0;
  const stepNoise = new Float32Array(latents.length);
  log("All three UNet sessions are resident. Beginning " + options.steps + " denoising steps." +
    (gpuScheduler ? " GPU scheduler avoids per-step latent readback." : ""));
  await yieldFrames();

  for (let step = startStep; step < options.steps; step++) {
    checkCancelled();
    if (infillData) {
      blendUnmaskedLatents(latents, sourceLatents, sourceNoise, latentMask, scheduler.sigmas[step]);
    }
    const stepStarted = performance.now();
    const scaled = gpuScheduler ? null : scheduler.scaleModelInput(latents, step);
    const denoiser = await runUnetStep(
      sessions,
      conditioning,
      scaled,
      scheduler.timesteps[step],
      options.width,
      options.height,
      options.cfg,
      step
    );
    totalUnetExecutionSeconds += denoiser.timings.a + denoiser.timings.b + denoiser.timings.c;
    if (infillData) {
      createNormalNoise(latentSize, options.seed + step + 1, stepNoise);
    } else {
      for (let image = 0; image < imageCount; image++) {
        createNormalNoise(
          latentSize,
          imageOptionsList[image].seed + step + 1,
          stepNoise.subarray(image * latentSize, (image + 1) * latentSize)
        );
      }
    }
    if (gpuScheduler) {
      gpuScheduler.step(stepNoise, scheduler.stepParameters(step, options.cfg, sessions.branchCount));
    } else {
      latents = scheduler.step(denoiser.guided, step, latents, stepNoise);
      for (let i = 0; i < latents.length; i++) {
        if (!Number.isFinite(latents[i])) throw new Error("Denoising produced a non-finite latent at " + i + ".");
      }
    }
    if (infillData) {
      blendUnmaskedLatents(latents, sourceLatents, sourceNoise, latentMask, scheduler.sigmas[step + 1]);
    }
    const percent = 8 + ((step + 1) / options.steps) * 82;
    setProgress(percent, "Denoising " + (step + 1) + " / " + options.steps);
    setStatus("Denoising step " + (step + 1) + " of " + options.steps + "…");
    log("Step " + (step + 1) + "/" + options.steps + " scheduler update " +
      ((performance.now() - stepStarted) / 1000).toFixed(2) + "s · sigma " +
      scheduler.sigmas[step].toFixed(5) + " → " + scheduler.sigmas[step + 1].toFixed(5));
    if ((step + 1) % 2 === 0 || step === options.steps - 1) await yieldFrames();
  }
  log("Accumulated UNet execution time " + totalUnetExecutionSeconds.toFixed(2) +
    "s across " + options.steps + " steps · " + imageCount + " image(s).");
  if (gpuScheduler) {
    latents = await gpuScheduler.readLatents();
    for (let i = 0; i < latents.length; i++) {
      if (!Number.isFinite(latents[i])) throw new Error("GPU scheduler produced a non-finite latent at " + i + ".");
    }
    log("GPU scheduler final latent readback complete · no per-step latent readback.");
  }

  checkCancelled();
  log("Denoising complete. Loading the transient VAE decoder for final image decode; UNet sessions remain resident.");
  setProgress(92, "Loading VAE decoder");
  setStatus("Loading VAE decoder…");
  const vaeSession = await ensureVaeSession(manifest);
  const canvases = [];
  try {
    for (let image = 0; image < imageCount; image++) {
      checkCancelled();
      const imageLatents = latents.subarray(image * latentSize, (image + 1) * latentSize);
      const scaledLatents = new Float16Array(latentSize);
      for (let i = 0; i < latentSize; i++) {
        scaledLatents[i] = imageLatents[i] / VAE_SCALING_FACTOR;
      }
      const inputName = vaeSession.inputNames[0];
      if (!inputName) throw new Error("VAE decoder graph has no input.");
      const inputIndex = vaeSession.inputNames.indexOf(inputName);
      validateStaticShape(vaeSession.inputMetadata?.[inputIndex],
        [1, 4, latentHeight, latentWidth], "VAE decoder input");
      log("Decoding image " + (image + 1) + "/" + imageCount + " · latent [1,4," +
        latentHeight + "," + latentWidth + "] · VAE scaling factor " + VAE_SCALING_FACTOR + ".");
      let outputs = null;
      try {
        const decodeStarted = performance.now();
        outputs = await vaeSession.run({
          [inputName]: new ort.Tensor("float16", scaledLatents, [1, 4, latentHeight, latentWidth])
        });
        const decodeSeconds = (performance.now() - decodeStarted) / 1000;
        const outputName = vaeSession.outputNames[0];
        if (!outputName || !outputs[outputName]) throw new Error("VAE decoder returned no image tensor.");
        const renderStarted = performance.now();
        let canvas = renderImage(outputs[outputName], options.width, options.height);
        if (infillData) {
          canvas = compositeInfillPixels(canvas, infillData.sourceCanvas, infillData.maskCanvas);
          log("Final pixel composite complete · unpainted source pixels restored exactly.");
        }
        const renderSeconds = (performance.now() - renderStarted) / 1000;
        log("Image " + (image + 1) + "/" + imageCount + " · VAE inference " +
          decodeSeconds.toFixed(2) + "s · pixel conversion " + renderSeconds.toFixed(2) +
          "s · combined " + (decodeSeconds + renderSeconds).toFixed(2) + "s.");
        canvases.push(canvas);
        if (onImage) await onImage(canvas, image);
      } finally {
        if (outputs) {
          for (const tensor of Object.values(outputs)) {
            if (tensor && typeof tensor.dispose === "function") tensor.dispose();
          }
        }
      }
      setProgress(92 + ((image + 1) / imageCount) * 8, "Decoded " + (image + 1) + " / " + imageCount);
      await yieldFrames();
    }
    setProgress(100, "Complete");
    return canvases;
  } finally {
    if (keepVaeResident) {
      log("VAE decoder retained in memory for the next image batch.");
    } else {
      await releaseVaeSession("decode batch complete; retain UNet for the next batch");
    }
  }
}

async function generate(options, infillData = null) {
  const canvases = await generateMany([options], infillData);
  return canvases[0];
}

prepareEngineButton.addEventListener("click", prepareInferenceEngine);

releaseEngineButton.addEventListener("click", async () => {
  if (button.disabled || prepareEngineButton.disabled) return;
  releaseEngineButton.disabled = true;
  button.disabled = true;
  prepareEngineButton.disabled = true;
  try {
    await releaseInferenceEngine();
    setStatus("Inference engine released. The downloaded model files remain cached.", "success");
    setProgress(0, "Engine released");
  } catch (error) {
    setStatus("Could not fully release inference engine: " + (error?.message || String(error)), "error");
    log("ERROR releasing engine: " + (error?.stack || error?.message || String(error)));
  } finally {
    button.disabled = false;
    prepareEngineButton.disabled = false;
    releaseEngineButton.disabled = !residentEngineLock;
  }
});

button.addEventListener("click", async () => {
  let options;
  try {
    options = readOptions();
  } catch (error) {
    setStatus(error.message || String(error), "error");
    return;
  }

  const batchCount = options.batchCount;
  const randomSeed = options.seed === -1;
  const baseSeed = options.seed;
  const completedSeeds = [];
  let completedCount = 0;

  button.disabled = true;
  prepareEngineButton.disabled = true;
  releaseEngineButton.disabled = true;
  cancelButton.disabled = false;
  cancelRequested = false;
  clearGeneratedResults();
  logOutput.textContent = "";
  progress.value = 0;
  activeBatch = null;
  setProgress(0, batchCount === 1 ? "Starting" : "Starting batch · " + batchCount + " images");
  generationStarted = performance.now();
  let lockAcquired = false;

  try {
    if (!residentEngineLock) {
      acquireHeavyModelTest("full Illustrious image generation");
      lockAcquired = true;
    }
    if (typeof Float16Array !== "function") throw new Error("This browser does not provide Float16Array.");
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("The browser did not return a WebGPU adapter.");
    if (ort.env.webgpu && !ort.env.webgpu.device) ort.env.webgpu.adapter = adapter;
    const adapterInfo = adapter.info || {};
    const adapterLabel = [adapterInfo.vendor, adapterInfo.architecture, adapterInfo.device, adapterInfo.description]
      .filter(Boolean).join(" · ");
    if (adapterLabel) log("WebGPU adapter identity · " + adapterLabel);
    log("WebGPU adapter ready · maxBufferSize " + formatBytes(adapter.limits?.maxBufferSize || 0) +
      " · maxStorageBufferBindingSize " + formatBytes(adapter.limits?.maxStorageBufferBindingSize || 0));
    const infillData = window.__illustriousInfill?.getPayload?.() || null;
    const parallelCount = infillData ? 1 : Math.min(options.parallelCount, batchCount);
    log("Starting batch · " + batchCount + " output image(s) · " + parallelCount +
      " image(s) per UNet pass · " +
      (randomSeed ? "a new random seed per image." : "starting seed " + baseSeed + " · increment by one per image."));

    for (let index = 0; index < batchCount; index += parallelCount) {
      checkCancelled();
      const actualCount = Math.min(parallelCount, batchCount - index);
      const imageOptions = [];
      for (let offset = 0; offset < parallelCount; offset++) {
        const seed = offset < actualCount
          ? (randomSeed ? resolveGenerationSeed(-1) : baseSeed + index + offset)
          : imageOptions[actualCount - 1].seed;
        imageOptions.push({ ...options, seed });
      }
      activeBatch = { index, count: batchCount, span: actualCount };
      for (let offset = 0; offset < actualCount; offset++) {
        log("Batch image " + (index + offset + 1) + "/" + batchCount + " · seed " + imageOptions[offset].seed + ".");
        if (randomSeed) log("Random seed selected · using seed " + imageOptions[offset].seed + ".");
      }
      if (actualCount < parallelCount) {
        log("Final GPU batch padded with " + (parallelCount - actualCount) +
          " duplicate latent(s) to keep the resident UNet shape; padded outputs are discarded.");
      }
      setProgress(0, "Starting " + actualCount + " image(s)");
      setStatus("Starting " + actualCount + " image(s) in one UNet pass…");

      await generateMany(imageOptions, infillData, async (canvas, localIndex) => {
        if (localIndex >= actualCount) return;
        const imageIndex = index + localIndex;
        const itemOptions = imageOptions[localIndex];
        const blob = await new Promise((resolve, reject) => canvas.toBlob(value =>
          value ? resolve(value) : reject(new Error("Could not encode image " + (imageIndex + 1) + " as PNG.")), "image/png"));
        appendGeneratedResult(blob, itemOptions, imageIndex, batchCount);
        completedSeeds.push(itemOptions.seed);
        completedCount++;
        log("Batch image " + (imageIndex + 1) + "/" + batchCount + " saved · PNG " + formatBytes(blob.size) + ".");
      });
      await yieldFrames();
    }

    activeBatch = null;
    setProgress(100, "Completed " + completedCount + " image(s)");
    setStatus("Generated " + completedCount + " image(s) · " + options.width + "×" + options.height +
      " · seeds " + completedSeeds.join(", ") + " · " +
      ((performance.now() - generationStarted) / 1000).toFixed(1) + "s.", "success");
    log("Batch finished successfully · " + completedCount + "/" + batchCount + " image(s).");
  } catch (error) {
    console.error("[Illustrious] Image batch failed:", error);
    const cancelled = cancelRequested || (error?.message || "").includes("cancelled");
    activeBatch = null;
    setProgress(batchCount ? completedCount / batchCount * 100 : 0,
      (cancelled ? "Cancelled" : "Stopped") + " after " + completedCount + "/" + batchCount + " image(s)");
    setStatus((cancelled ? "Batch cancelled" : "Batch stopped after an error") +
      " · " + completedCount + "/" + batchCount + " image(s) completed. " +
      (error?.message || String(error)), cancelled ? "info" : "error");
    log((cancelled ? "Batch cancelled: " : "BATCH ERROR: ") +
      (error?.stack || error?.message || String(error)));
  } finally {
    activeBatch = null;
    if (lockAcquired) releaseHeavyModelTest("full Illustrious image generation");
    button.disabled = false;
    prepareEngineButton.disabled = false;
    releaseEngineButton.disabled = !residentEngineLock;
    cancelButton.disabled = true;
  }
});

fields.lora.addEventListener("change", () => {
  fields.loraStrength.disabled = !fields.lora.checked;
});

settingsExportButton?.addEventListener("click", () => {
  try {
    downloadGenerationSettings();
  } catch (error) {
    settingsStatus.textContent = "Settings export failed: " + (error?.message || String(error));
  }
});

settingsImportButton?.addEventListener("click", () => settingsImportInput?.click());
settingsImportInput?.addEventListener("change", async () => {
  const file = settingsImportInput.files?.[0];
  if (!file) return;
  try {
    await importGenerationSettings(file);
  } catch (error) {
    settingsStatus.textContent = "Settings import failed: " + (error?.message || String(error));
  } finally {
    settingsImportInput.value = "";
  }
});

cancelButton.addEventListener("click", () => {
  cancelRequested = true;
  cancelButton.disabled = true;
  setStatus("Cancellation requested. The current WebGPU graph call must finish before stopping…");
  log("Cancellation requested; the current ONNX graph invocation is not interruptible.");
});


const cacheStatus = document.querySelector("#modelCacheStatus");
const importBundleInput = document.querySelector("#importModelBundleFile");
document.querySelector("#prepareModelCache")?.addEventListener("click", async () => {
  try {
    const manifest = await fetchJSON(MANIFEST_URL, "Model manifest", true);
    const result = await prepareModelCache(manifest, message => {
      cacheStatus.textContent = message;
      console.log("[Model cache] " + message);
    }, (value, path) => {
      cacheStatus.textContent = "Caching " + Math.round(value) + "% · " + path;
    }, () => {}, true);
    cacheStatus.textContent = "Ready · " + result.totalFiles + " files including VAE encoder and scheduler · " +
      (result.bytes / 1024 ** 3).toFixed(2) + " GiB · " + result.reused + " reused.";
  } catch (error) {
    cacheStatus.textContent = "Cache preparation failed: " + error.message;
  }
});
document.querySelector("#exportModelBundle")?.addEventListener("click", async () => {
  try {
    const manifest = await fetchJSON(MANIFEST_URL, "Model manifest");
    cacheStatus.textContent = "Building portable model bundle…";
    const blob = await exportModelBundle(manifest, message => cacheStatus.textContent = message);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "wai-illustrious-v17-browser-model.wkimodel";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
    cacheStatus.textContent = "Bundle ready · " + (blob.size / 1024 ** 3).toFixed(2) + " GiB.";
  } catch (error) {
    cacheStatus.textContent = "Bundle export failed: " + error.message;
  }
});
document.querySelector("#importModelBundle")?.addEventListener("click", () => importBundleInput?.click());
importBundleInput?.addEventListener("change", async () => {
  const file = importBundleInput.files?.[0];
  if (!file) return;
  try {
    const manifest = await fetchJSON(MANIFEST_URL, "Model manifest", true);
    cacheStatus.textContent = "Importing and verifying model bundle…";
    const count = await importModelBundle(file, manifest, message => cacheStatus.textContent = message,
      value => cacheStatus.textContent = "Importing " + Math.round(value) + "%…");
    cacheStatus.textContent = "Imported and verified " + count + " files. Ready to generate.";
  } catch (error) {
    cacheStatus.textContent = "Bundle import failed: " + error.message;
  } finally {
    importBundleInput.value = "";
  }
});
