import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";
import { CLIPTokenizer } from "../vendor/transformers/transformers.min.js";
import { acquireHeavyModelTest, releaseHeavyModelTest } from "./heavy-model-test-lock.js";
import { preparePersonalAmiComponent, releasePersonalAmiPayload } from "./personal-ami-lora.js";
import { fetchModelResource, prepareModelCache, exportModelBundle, importModelBundle } from "./model-cache.js";

const MODEL_ID = "aaaaaeeeee/blip-sdxl-anime-wai-illustrious17";
const MODEL_BASE = "https://huggingface.co/" + MODEL_ID + "/resolve/main/";
const MANIFEST_URL = MODEL_BASE + "manifest.json";
const SCHEDULER_URL = "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/b41f1c324e32e03f9e104bd9ace0992e4e135597/scheduler/scheduler_config.json";
const VAE_SCALING_FACTOR = 0.18215; // The WAI pack's Kohaku XL beta5 base VAE config.

ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

const button = document.querySelector("#generateImage");
const cancelButton = document.querySelector("#cancelGeneration");
const status = document.querySelector("#generationStatus");
const progress = document.querySelector("#generationProgress");
const progressLabel = document.querySelector("#generationProgressLabel");
const logOutput = document.querySelector("#generationLog");
const image = document.querySelector("#generatedImage");
const downloadLink = document.querySelector("#downloadGeneratedImage");
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
  personalAmi: document.querySelector("#generationPersonalAmi"),
  loraStrength: document.querySelector("#generationLoraStrength"),
  loraFile: document.querySelector("#loraFile")
};

const adapterFileLabel = document.querySelector("#fileName");
fields.loraFile?.addEventListener("change", () => {
  const file = fields.loraFile.files?.[0];
  if (!adapterFileLabel) return;
  adapterFileLabel.textContent = file
    ? file.name + " · " + (file.size / 1024 ** 2).toFixed(1) + " MiB"
    : "No adapter selected";
});

let cancelRequested = false;
let generationStarted = 0;
let imageObjectUrl = null;
let residentUnetSessions = null;
let residentUnetKey = null;
let residentVaeSession = null;
let residentEngineLock = false;
const conditioningCache = new Map();

function setStatus(message, type = "info") {
  status.textContent = message;
  status.dataset.type = type;
}

function setProgress(value, label) {
  progress.value = value;
  progressLabel.textContent = label;
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

async function fetchJSON(url, label) {
  const response = await fetchModelResource(url, { credentials: "omit" });
  if (!response.ok) throw new Error(label + " request failed: HTTP " + response.status);
  return response.json();
}

async function loadTokenizer(subfolder) {
  const folderURL = MODEL_BASE + subfolder + "/";

  async function readJSON(filename) {
    const url = folderURL + filename;
    const response = await fetchModelResource(url, { credentials: "omit" });
    if (!response.ok) throw new Error("Tokenizer request failed: " + filename + " · HTTP " + response.status);
    return response.json();
  }

  async function readText(filename) {
    const url = folderURL + filename;
    const response = await fetch(url, { credentials: "omit" });
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

function tensorAsFloat32(tensor, label) {
  if (!tensor?.data || !Array.isArray(tensor.dims)) throw new Error(label + " has no readable tensor data.");
  let source;
  if (tensor.type === "float16") {
    source = tensor.data instanceof Float16Array
      ? tensor.data
      : new Float16Array(tensor.data.buffer.slice(
        tensor.data.byteOffset, tensor.data.byteOffset + tensor.data.byteLength
      ));
  } else if (tensor.type === "float32") {
    source = tensor.data instanceof Float32Array ? tensor.data : Float32Array.from(tensor.data);
  } else {
    throw new Error(label + " has unsupported dtype " + tensor.type + ".");
  }
  const output = new Float32Array(source.length);
  for (let i = 0; i < source.length; i++) {
    const value = Number(source[i]);
    if (!Number.isFinite(value)) throw new Error(label + " contains a non-finite value at " + i + ".");
    output[i] = value;
  }
  return output;
}

function externalData(component) {
  if (!component?.onnx || !Array.isArray(component.shards) || !component.shards.length) {
    throw new Error("Model manifest component is missing its ONNX graph or external-data shards.");
  }
  return component.shards.map(path => ({
    path: path.slice(path.lastIndexOf("/") + 1),
    data: MODEL_BASE + path
  }));
}

async function loadSession(manifest, key, label, patchedExternalData = null, preferredOutputLocation = null) {
  const component = manifest?.components?.[key];
  const started = performance.now();
  log("Loading " + label + " (" + formatBytes(component?.bytes) + ")…");
  const sessionOptions = {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "all",
    externalData: patchedExternalData || externalData(component)
  };
  if (preferredOutputLocation) sessionOptions.preferredOutputLocation = preferredOutputLocation;
  const session = await ort.InferenceSession.create(MODEL_BASE + component.onnx, sessionOptions);
  log(label + " session created in " + ((performance.now() - started) / 1000).toFixed(2) +
    "s · inputs: " + session.inputNames.join(", ") +
    (preferredOutputLocation ? " · GPU-resident outputs enabled" : ""));
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

async function encodePrompts(manifest, prompt, negativePrompt, personalAmi) {
  const options = { padding: "max_length", truncation: true, max_length: 77 };
  log("Loading CLIP-L and tokenizing positive/negative prompts…");
  const tokenizerL = await loadTokenizer("tokenizer");
  const encodedL = await tokenizerL(prompt, options);
  const negativeEncodedL = await tokenizerL(negativePrompt, options);
  const idsL = tokenIds(encodedL, "CLIP-L positive");
  const negativeIdsL = tokenIds(negativeEncodedL, "CLIP-L negative");
  const startedL = performance.now();
  let sessionL;
  let hiddenL;
  let negativeHiddenL;
  try {
    const patchedL = personalAmi
      ? await preparePersonalAmiComponent(personalAmi.file, manifest, "text_encoder", personalAmi.strength, log, checkCancelled)
      : null;
    sessionL = await loadSession(manifest, "text_encoder", "CLIP-L", patchedL?.externalData);
    if (canBatchEncoder(sessionL, "input_ids")) {
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
  log("CLIP-L positive + negative encoded in " + ((performance.now() - startedL) / 1000).toFixed(2) + "s; released.");

  log("Loading CLIP-bigG and tokenizing positive/negative prompts…");
  const tokenizerG = await loadTokenizer("tokenizer_2");
  const encodedG = await tokenizerG(prompt, options);
  const negativeEncodedG = await tokenizerG(negativePrompt, options);
  const idsG = tokenIds(encodedG, "CLIP-bigG positive");
  const negativeIdsG = tokenIds(negativeEncodedG, "CLIP-bigG negative");
  const startedG = performance.now();
  let sessionG;
  let hiddenG;
  let negativeHiddenG;
  let pooledG;
  let negativePooledG;
  try {
    const patchedG = personalAmi
      ? await preparePersonalAmiComponent(personalAmi.file, manifest, "text_encoder_2", personalAmi.strength, log, checkCancelled)
      : null;
    sessionG = await loadSession(manifest, "text_encoder_2", "CLIP-bigG", patchedG?.externalData);
    if (canBatchEncoder(sessionG, "input_ids")) {
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
  log("CLIP-bigG positive + negative encoded in " + ((performance.now() - startedG) / 1000).toFixed(2) + "s; released.");

  const positive = combineHidden(hiddenL, hiddenG, "Positive conditioning");
  const negative = combineHidden(negativeHiddenL, negativeHiddenG, "Negative conditioning");
  const encoderHidden = new Float16Array(2 * 77 * 2048);
  encoderHidden.set(negative, 0);
  encoderHidden.set(positive, negative.length);
  const textEmbeds = new Float16Array(2 * 1280);
  textEmbeds.set(negativePooledG, 0);
  textEmbeds.set(pooledG, 1280);
  log("Classifier-free conditioning ready · encoder_hidden_states [2,77,2048] · text_embeds [2,1280].");
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

function createNormalNoise(length, seed) {
  const random = createRng(seed);
  const output = new Float32Array(length);
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

function makeBatchSample(scaled, latentHeight, latentWidth, batchSize = 2) {
  const oneSize = 4 * latentHeight * latentWidth;
  const batch = new Float16Array(oneSize * batchSize);
  batch.set(scaled, 0);
  if (batchSize === 2) batch.set(scaled, oneSize);
  return batch;
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

function createGpuInput(device, dataType, data, dims, label) {
  const buffer = device.createBuffer({
    size: Math.max(4, data.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
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

function createGpuInputState(conditioning, width, height, batchSize, conditioningIndex) {
  const device = ort.env.webgpu?.device;
  if (!device || typeof ort.Tensor.fromGpuBuffer !== "function") {
    throw new Error("ONNX Runtime did not expose the GPUDevice/Tensor.fromGpuBuffer API required for reusable GPU inputs.");
  }
  const latentWidth = width / 8;
  const latentHeight = height / 8;
  const latentLength = 4 * latentHeight * latentWidth;
  const hiddenLength = 77 * 2048;
  const pooledLength = 1280;
  const hiddenInput = batchSize === 1
    ? conditioning.encoderHidden.subarray(conditioningIndex * hiddenLength, (conditioningIndex + 1) * hiddenLength)
    : conditioning.encoderHidden;
  const pooledInput = batchSize === 1
    ? conditioning.textEmbeds.subarray(conditioningIndex * pooledLength, (conditioningIndex + 1) * pooledLength)
    : conditioning.textEmbeds;
  const entries = {};
  try {
    entries.sample = createGpuInput(device, "float16", new Float16Array(latentLength * batchSize),
      [batchSize, 4, latentHeight, latentWidth], "UNET_A sample");
    entries.timestep = createGpuInput(device, "float32", new Float32Array(batchSize),
      [batchSize], "UNET_A timestep");
    entries.encoderHidden = createGpuInput(device, "float16", hiddenInput,
      [batchSize, 77, 2048], "shared encoder_hidden_states");
    entries.textEmbeds = createGpuInput(device, "float16", pooledInput,
      [batchSize, 1280], "UNET_A text_embeds");
    entries.timeIds = createGpuInput(device, "float32", makeBatchTimeIds(height, width, batchSize),
      [batchSize, 6], "UNET_A time_ids");
    entries.cOutput = createGpuInput(device, "float16",
      new Float16Array(latentLength * batchSize), [batchSize, 4, latentHeight, latentWidth],
      "UNET_C reusable output");
  } catch (error) {
    for (const entry of Object.values(entries)) entry.buffer.destroy();
    throw error;
  }
  return {
    device,
    entries,
    batchSize,
    writeStep(sample, timestep) {
      if (sample.byteLength !== entries.sample.buffer.size) {
        throw new Error("Latent input size changed while the UNet engine was resident.");
      }
      entries.timestep.data.fill(timestep);
      device.queue.writeBuffer(entries.sample.buffer, 0, sample);
      device.queue.writeBuffer(entries.timestep.buffer, 0, entries.timestep.data);
    },
    updateConditioning(next, nextConditioningIndex = conditioningIndex) {
      const hidden = batchSize === 1
        ? next.encoderHidden.subarray(nextConditioningIndex * hiddenLength, (nextConditioningIndex + 1) * hiddenLength)
        : next.encoderHidden;
      const pooled = batchSize === 1
        ? next.textEmbeds.subarray(nextConditioningIndex * pooledLength, (nextConditioningIndex + 1) * pooledLength)
        : next.textEmbeds;
      if (hidden.byteLength !== entries.encoderHidden.buffer.size ||
          pooled.byteLength !== entries.textEmbeds.buffer.size) {
        throw new Error("Prompt conditioning dimensions changed while the UNet engine was resident.");
      }
      device.queue.writeBuffer(entries.encoderHidden.buffer, 0, hidden);
      device.queue.writeBuffer(entries.textEmbeds.buffer, 0, pooled);
    },
    destroy() {
      for (const entry of Object.values(entries)) {
        if (entry?.buffer) entry.buffer.destroy();
      }
    }
  };
}


async function createGpuScheduler(device, gpuInputs, width, height, batchSize) {
  const latentSize = 4 * (height / 8) * (width / 8);
  const latentBuffer = device.createBuffer({
    size: latentSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  });
  const noiseBuffer = device.createBuffer({
    size: latentSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  const uniformBuffer = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  const shader = device.createShaderModule({ code: `
    struct Params {
      count: u32,
      batch: u32,
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
      if (params.batch == 2u) {
        let conditional0 = halfAt(params.latentSize + first);
        let conditional1 = halfAt(params.latentSize + second);
        epsilon0 = epsilon0 + params.cfg * (conditional0 - epsilon0);
        epsilon1 = epsilon1 + params.cfg * (conditional1 - epsilon1);
      }

      let latent0 = latents[first] + epsilon0 * params.dt + noise[first] * params.sigmaUp;
      let latent1 = latents[second] + epsilon1 * params.dt + noise[second] * params.sigmaUp;
      latents[first] = latent0;
      latents[second] = latent1;

      let packed = pack2x16float(vec2<f32>(latent0 * params.nextScale, latent1 * params.nextScale));
      nextSample[pair] = packed;
      if (params.batch == 2u) {
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
    return {
      latentBuffer, noiseBuffer, uniformBuffer, pipeline, bindGroup, latentSize, batchSize,
      initialize(latents, scaledLatents) {
        if (latents.length !== latentSize || scaledLatents.length !== latentSize) {
          throw new Error("Initial GPU scheduler latent dimensions do not match the selected resolution.");
        }
        device.queue.writeBuffer(latentBuffer, 0, latents);
        const batchSample = makeBatchSample(scaledLatents, height / 8, width / 8, batchSize);
        device.queue.writeBuffer(gpuInputs.entries.sample.buffer, 0, batchSample);
      },
      step(noise, stepParams) {
        if (noise.length !== latentSize) throw new Error("GPU scheduler noise dimensions changed.");
        device.queue.writeBuffer(noiseBuffer, 0, noise);
        const raw = new ArrayBuffer(32);
        const view = new DataView(raw);
        view.setUint32(0, latentSize, true);
        view.setUint32(4, batchSize, true);
        view.setFloat32(8, stepParams.cfg, true);
        view.setFloat32(12, stepParams.dt, true);
        view.setFloat32(16, stepParams.sigmaUp, true);
        view.setFloat32(20, stepParams.nextScale, true);
        view.setUint32(24, latentSize, true);
        view.setUint32(28, 0, true);
        device.queue.writeBuffer(uniformBuffer, 0, raw);
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(latentSize / 2 / 256));
        pass.end();
        device.queue.submit([encoder.finish()]);
      },
      async readLatents() {
        const readback = device.createBuffer({
          size: latentSize * 4,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(latentBuffer, 0, readback, 0, latentSize * 4);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        readback.destroy();
        return result;
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
  const batchSample = sessions.gpuScheduler
    ? null
    : makeBatchSample(scaledLatents, latentHeight, latentWidth, batch);
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

  try {
    if (!sessions.gpuScheduler) gpu.writeStep(batchSample, timestep);
    const startedA = performance.now();
    outputsA = await sessions.a.run({
      sample: gpu.entries.sample.tensor,
      timestep: gpu.entries.timestep.tensor,
      encoder_hidden_states: gpu.entries.encoderHidden.tensor,
      text_embeds: gpu.entries.textEmbeds.tensor,
      time_ids: gpu.entries.timeIds.tensor
    });
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
    });
    timings.b = (performance.now() - startedB) / 1000;
    checkCancelled();
    validateGpuTensor(outputsB.hidden_out, "UNET_B hidden_out",
      shape(1280, latentHeight / 2, latentWidth / 2));

    for (const name of ["hidden", "skip_6", "skip_7", "skip_8"]) {
      disposeOutput(outputsA, name);
    }

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
      guided = new Float32Array(latentSize);
      if (batch === 1) {
        for (let i = 0; i < latentSize; i++) guided[i] = noisePrediction[i];
      } else {
        for (let i = 0; i < latentSize; i++) {
          const unconditional = noisePrediction[i];
          const conditional = noisePrediction[latentSize + i];
          guided[i] = unconditional + cfg * (conditional - unconditional);
        }
      }
    }

    log("Step denoiser timing · A " + timings.a.toFixed(2) + "s · B " +
      timings.b.toFixed(2) + "s · C " + timings.c.toFixed(2) +
      "s · CFG " + cfg + " · batch " + batch + " · A→B→C GPU-resident · reusable GPU inputs · standard dispatch");
    return { guided, timings };
  } finally {
    disposeRemaining(outputsC);
    disposeRemaining(outputsB);
    disposeRemaining(outputsA);
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
  const values = tensorAsFloat32(tensor, "VAE decoded image");
  if (tensor.dims.length !== 4 || tensor.dims[0] !== 1 ||
      tensor.dims[1] !== 3 || tensor.dims[2] !== height || tensor.dims[3] !== width) {
    throw new Error("VAE output shape [" + tensor.dims.join(", ") +
      "] does not match expected [1, 3, " + height + ", " + width + "].");
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: false });
  if (!context) throw new Error("Could not create a 2D canvas context.");
  const pixels = context.createImageData(width, height);
  const planeSize = width * height;
  for (let pixel = 0; pixel < planeSize; pixel++) {
    const r = Math.max(0, Math.min(255, Math.round((values[pixel] * 0.5 + 0.5) * 255)));
    const g = Math.max(0, Math.min(255, Math.round((values[planeSize + pixel] * 0.5 + 0.5) * 255)));
    const b = Math.max(0, Math.min(255, Math.round((values[planeSize * 2 + pixel] * 0.5 + 0.5) * 255)));
    const offset = pixel * 4;
    pixels.data[offset] = r;
    pixels.data[offset + 1] = g;
    pixels.data[offset + 2] = b;
    pixels.data[offset + 3] = 255;
  }
  context.putImageData(pixels, 0, 0);
  return canvas;
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
  if (!prompt) throw new Error("Enter a prompt first.");
  if (!Number.isInteger(width) || width < 512 || width > 1024 || width % 64 !== 0) {
    throw new Error("Width must be 512–1024 and divisible by 64.");
  }
  if (!Number.isInteger(height) || height < 512 || height > 1024 || height % 64 !== 0) {
    throw new Error("Height must be 512–1024 and divisible by 64.");
  }
  if (!Number.isInteger(steps) || steps < 1 || steps > 100) throw new Error("Steps must be between 1 and 100.");
  if (!Number.isFinite(cfg) || cfg < 0 || cfg > 20) throw new Error("CFG scale must be between 0 and 20.");
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 4294967295) {
    throw new Error("Seed must be an integer from 0 to 4294967295.");
  }
  if (fields.personalAmi.checked) {
    if (!fields.loraFile.files?.[0]) throw new Error("Choose the Personal Ami .safetensors file in the adapter inspector above.");
    const strength = Number(fields.loraStrength.value);
    if (!Number.isFinite(strength) || strength < 0 || strength > 2) {
      throw new Error("Personal Ami strength must be between 0 and 2.");
    }
  }
  const adapterPrompt = fields.personalAmi.checked && !/\\bpersonal\\s+ami\\b/i.test(prompt)
    ? prompt + ", personal ami"
    : prompt;
  return {
    prompt: adapterPrompt + (suffix ? ", " + suffix : ""),
    negative,
    width,
    height,
    steps,
    cfg,
    seed,
    personalAmi: fields.personalAmi.checked ? {
      file: fields.loraFile.files?.[0] || null,
      strength: Number(fields.loraStrength.value)
    } : null
  };
}


function adapterCacheKey(personalAmi) {
  if (!personalAmi) return "base";
  const file = personalAmi.file;
  return [
    file.name,
    file.size,
    file.lastModified,
    personalAmi.strength
  ].join("|");
}

function conditioningCacheKey(options) {
  return JSON.stringify([
    options.prompt,
    options.negative,
    adapterCacheKey(options.personalAmi)
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
    options.personalAmi
  );
  conditioningCache.set(key, conditioning);
  while (conditioningCache.size > 8) {
    conditioningCache.delete(conditioningCache.keys().next().value);
  }
  log("Cached prompt conditioning · " + conditioningCache.size + "/8 entries.");
  return conditioning;
}

async function buildUnetSessions(manifest, personalAmi, conditioning, width, height, batchSize, conditioningIndex, patchCache) {
  const sessions = { a: null, b: null, c: null, gpuInputs: null, batchSize };
  const patchedComponent = async name => {
    if (!personalAmi) return null;
    if (!patchCache.has(name)) {
      patchCache.set(name, await preparePersonalAmiComponent(
        personalAmi.file, manifest, name, personalAmi.strength, log, checkCancelled
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
    sessions.a = await loadSession(manifest, "unet_a", "UNET_A", patchedA?.externalData, outputsA);
    sessions.gpuInputs = createGpuInputState(conditioning, width, height, batchSize, conditioningIndex);
    checkCancelled();

    const patchedB = await patchedComponent("unet_b");
    sessions.b = await loadSession(manifest, "unet_b", "UNET_B", patchedB?.externalData, outputsB);
    checkCancelled();

    const patchedC = await patchedComponent("unet_c");
    try {
      sessions.gpuScheduler = await createGpuScheduler(
        ort.env.webgpu.device, sessions.gpuInputs, width, height, batchSize
      );
      log("GPU-resident Euler scheduler initialized · CFG and latent update stay on WebGPU.");
    } catch (error) {
      sessions.gpuScheduler = null;
      sessions.gpuInputs.entries.cOutput.buffer.destroy();
      sessions.gpuInputs.entries.cOutput = null;
      log("GPU scheduler unavailable; retaining the CPU scheduler. Reason: " +
        (error?.message || String(error)));
    }
    sessions.c = await loadSession(
      manifest, "unet_c", "UNET_C", patchedC?.externalData,
      sessions.gpuScheduler ? "gpu-buffer" : null
    );

    validateSessionInputs(sessions.a, ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"], "UNET_A");
    validateSessionInputs(sessions.b, ["hidden", "emb", "encoder_hidden_states", "skip_6", "skip_7", "skip_8"], "UNET_B");
    validateSessionInputs(sessions.c, ["hidden", "emb", "encoder_hidden_states", "skip_0", "skip_1", "skip_2", "skip_3", "skip_4", "skip_5"], "UNET_C");
    return sessions;
  } catch (error) {
    await releaseSessions(sessions);
    throw error;
  }
}

async function ensureUnetSessions(manifest, personalAmi, conditioning, width, height, cfg) {
  const batchSize = cfg === 0 || cfg === 1 ? 1 : 2;
  const conditioningIndex = cfg === 1 ? 1 : 0;
  const key = adapterCacheKey(personalAmi) + "|" + width + "x" + height + "|batch" + batchSize;
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

  setProgress(8, "Preparing UNet partitions");
  setStatus("Preparing UNET_A, UNET_B, and UNET_C. This requires several GiB of memory…");
  const patchCache = new Map();
  let sessions;
  try {
    sessions = await buildUnetSessions(
      manifest, personalAmi, conditioning, width, height, batchSize, conditioningIndex, patchCache
    );
    log("UNET_A/B/C ready · standard WebGPU dispatch with persistent GPU inputs and GPU-resident partition handoffs." +
      (batchSize === 1 ? " Single-branch CFG path · batch=1." : " Classifier-free guidance path · batch=2."));

    if (personalAmi) {
      releasePersonalAmiPayload(personalAmi.file);
      log("Released the temporary in-memory LoRA factor copy after all five components were patched.");
    }
    residentUnetSessions = sessions;
    residentUnetKey = key;
    releaseEngineButton.disabled = false;
    log("UNet engine prepared · all three sessions remain resident for subsequent images. No warm-up inference required.");

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
    log("WebGPU adapter ready · maxBufferSize " + formatBytes(adapter.limits?.maxBufferSize || 0) +
      " · maxStorageBufferBindingSize " + formatBytes(adapter.limits?.maxStorageBufferBindingSize || 0));

    const manifest = await fetchJSON(MANIFEST_URL, "Model manifest");
    setProgress(1, "Preparing model downloads");
    setStatus("Downloading and verifying model files into persistent browser cache…");
    const cacheResult = await prepareModelCache(manifest, log,
      (value, path) => setProgress(1 + value * 5, "Caching model files · " + path),
      checkCancelled);
    log("Preload complete · " + cacheResult.downloaded + " new downloads, " +
      cacheResult.reused + " cached files.");

    setStatus("Precomputing prompt conditioning…");
    setProgress(7, "Precomputing prompt conditioning");
    const conditioning = await getConditioning(manifest, options);
    checkCancelled();

    setStatus("Preparing resident UNet sessions…");
    await ensureUnetSessions(manifest, options.personalAmi, conditioning, options.width, options.height, options.cfg);
    checkCancelled();

    setStatus("Preparing resident VAE decoder…");
    setProgress(96, "Preparing VAE decoder");
    await ensureVaeSession(manifest);
    setProgress(100, "Engine ready");
    setStatus("Inference engine prepared · UNet and VAE sessions are resident; prompt conditioning is cached.", "success");
    log("Ready for repeated generation. Change prompt/adapter settings as needed; matching conditioning and model sessions will be reused.");
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

async function generate(options) {
  const manifest = await fetchJSON(MANIFEST_URL, "Model manifest");
  setProgress(1, "Preparing model downloads");
  setStatus("Downloading and verifying model files into persistent browser cache…");
  const cacheResult = await prepareModelCache(manifest, log, (value, path) => setProgress(1 + value * 5, "Caching model files · " + path), checkCancelled);
  log("Preload complete · " + cacheResult.downloaded + " new downloads, " + cacheResult.reused + " cached files.");
  const schedulerConfig = await fetchJSON(SCHEDULER_URL, "Scheduler config");
  const latentWidth = options.width / 8;
  const latentHeight = options.height / 8;
  const latentSize = 4 * latentWidth * latentHeight;
  const rngNoiseSeeds = options.seed;

  log("Model " + (manifest.packageId || MODEL_ID) + " · WebGPU · " +
    options.width + "×" + options.height + " · " + options.steps + " steps.");
  log("Prompt: " + options.prompt);
  log("Negative: " + (options.negative || "(empty)"));
  log("Seed " + options.seed + " · CFG " + options.cfg);
  log("CLIP positive/negative prompts are batched when supported. CFG 0 or 1 uses one UNet branch (batch 1); other CFG values use both branches (batch 2). Reusable GPU inputs and A→B→C GPU-resident handoffs are enabled.");
  if (options.personalAmi) {
    log("Personal Ami enabled · " + options.personalAmi.file.name + " · strength " + options.personalAmi.strength.toFixed(3) + " · all 980 mapped pairs will be patched.");
  } else {
    log("Personal Ami disabled · using unmodified WAI Illustrious v17 base weights.");
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
  let latents = createNormalNoise(latentSize, rngNoiseSeeds);
  for (let i = 0; i < latents.length; i++) latents[i] *= scheduler.initNoiseSigma;

  const parts = ["unet_a", "unet_b", "unet_c"];
  const unetBytes = parts.reduce((sum, key) => sum + Number(manifest.components?.[key]?.bytes || 0), 0);
  log("UNet manifest payload total " + formatBytes(unetBytes) +
    " (not a direct VRAM measurement). Loading all partitions together…");
  const sessions = await ensureUnetSessions(manifest, options.personalAmi, conditioning, options.width, options.height, options.cfg);
  const gpuScheduler = sessions.gpuScheduler;
  if (gpuScheduler) {
    const initialScaled = scheduler.scaleModelInput(latents, 0);
    gpuScheduler.initialize(latents, new Float16Array(initialScaled));
  }
  let totalUnetExecutionSeconds = 0;
  log("All three UNet sessions are resident. Beginning " + options.steps + " denoising steps." +
    (gpuScheduler ? " GPU scheduler avoids per-step latent readback." : ""));
  await yieldFrames();

  for (let step = 0; step < options.steps; step++) {
    checkCancelled();
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
    const noise = createNormalNoise(latentSize, options.seed + step + 1);
    if (gpuScheduler) {
      gpuScheduler.step(noise, scheduler.stepParameters(step, options.cfg, sessions.batchSize));
    } else {
      latents = scheduler.step(denoiser.guided, step, latents, noise);
      for (let i = 0; i < latents.length; i++) {
        if (!Number.isFinite(latents[i])) throw new Error("Denoising produced a non-finite latent at " + i + ".");
      }
    }
    const percent = 8 + ((step + 1) / options.steps) * 82;
    setProgress(percent, "Denoising " + (step + 1) + " / " + options.steps);
    setStatus("Denoising step " + (step + 1) + " of " + options.steps + "…");
    log("Step " + (step + 1) + "/" + options.steps + " scheduler update " +
      ((performance.now() - stepStarted) / 1000).toFixed(2) + "s · sigma " +
      scheduler.sigmas[step].toFixed(5) + " → " + scheduler.sigmas[step + 1].toFixed(5));
    await yieldFrames();
  }
  log("Accumulated UNet execution time " + totalUnetExecutionSeconds.toFixed(2) +
    "s across " + options.steps + " steps (sum of A + B + C inference calls).");
  if (gpuScheduler) {
    latents = await gpuScheduler.readLatents();
    for (let i = 0; i < latents.length; i++) {
      if (!Number.isFinite(latents[i])) throw new Error("GPU scheduler produced a non-finite latent at " + i + ".");
    }
    log("GPU scheduler final latent readback complete · no per-step latent readback.");
  }

  checkCancelled();
  log("Denoising complete. Loading VAE decoder after releasing all UNet partitions.");
  setProgress(92, "Loading VAE decoder");
  setStatus("Loading VAE decoder…");
  const vaeSession = await ensureVaeSession(manifest);
  checkCancelled();
  const scaledLatents = new Float16Array(latents.length);
  for (let i = 0; i < latents.length; i++) {
    scaledLatents[i] = latents[i] / VAE_SCALING_FACTOR;
  }
  const inputName = vaeSession.inputNames[0];
  if (!inputName) throw new Error("VAE decoder graph has no input.");
  const inputIndex = vaeSession.inputNames.indexOf(inputName);
  validateStaticShape(vaeSession.inputMetadata?.[inputIndex],
    [1, 4, latentHeight, latentWidth], "VAE decoder input");
  const started = performance.now();
  log("Decoding latent [1,4," + latentHeight + "," + latentWidth +
    "] · VAE scaling factor " + VAE_SCALING_FACTOR + ".");
  const outputs = await vaeSession.run({
    [inputName]: new ort.Tensor("float16", scaledLatents, [1, 4, latentHeight, latentWidth])
  });
  const outputName = vaeSession.outputNames[0];
  if (!outputName || !outputs[outputName]) throw new Error("VAE decoder returned no image tensor.");
  const canvas = renderImage(outputs[outputName], options.width, options.height);
  log("VAE decode + canvas conversion " + ((performance.now() - started) / 1000).toFixed(2) + "s.");
  setProgress(100, "Complete");
  return canvas;
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

  button.disabled = true;
  prepareEngineButton.disabled = true;
  releaseEngineButton.disabled = true;
  cancelButton.disabled = false;
  cancelRequested = false;
  logOutput.textContent = "";
  progress.value = 0;
  setProgress(0, "Starting");
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
    log("WebGPU adapter ready · maxBufferSize " + formatBytes(adapter.limits?.maxBufferSize || 0) +
      " · maxStorageBufferBindingSize " + formatBytes(adapter.limits?.maxStorageBufferBindingSize || 0));
    setStatus("Starting browser-side inference…");
    const canvas = await generate(options);
    if (cancelRequested) throw new Error("Generation cancelled by user.");
    const blob = await new Promise((resolve, reject) => canvas.toBlob(value =>
      value ? resolve(value) : reject(new Error("Could not encode the generated canvas as PNG.")), "image/png"));
    if (imageObjectUrl) URL.revokeObjectURL(imageObjectUrl);
    imageObjectUrl = URL.createObjectURL(blob);
    image.src = imageObjectUrl;
    image.hidden = false;
    downloadLink.href = imageObjectUrl;
    downloadLink.download = "illustrious-" + options.seed + ".png";
    downloadLink.hidden = false;
    setStatus("Image generated · " + options.width + "×" + options.height +
      " · seed " + options.seed + " · " + ((performance.now() - generationStarted) / 1000).toFixed(1) + "s.", "success");
    log("Generation finished successfully. PNG size " + formatBytes(blob.size) + ".");
  } catch (error) {
    console.error("[Illustrious] Full generation failed:", error);
    const cancelled = cancelRequested || (error?.message || "").includes("cancelled");
    setStatus((cancelled ? "Cancelled: " : "Generation failed: ") + (error?.message || String(error)), cancelled ? "info" : "error");
    log((cancelled ? "Cancelled: " : "ERROR: ") + (error?.stack || error?.message || String(error)));
  } finally {
    if (lockAcquired) releaseHeavyModelTest("full Illustrious image generation");
    button.disabled = false;
    prepareEngineButton.disabled = false;
    releaseEngineButton.disabled = !residentEngineLock;
    cancelButton.disabled = true;
    if (progress.value < 100 && cancelRequested) setProgress(progress.value, "Cancelled");
  }
});

fields.personalAmi.addEventListener("change", () => {
  fields.loraStrength.disabled = !fields.personalAmi.checked;
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
    const manifest = await fetchJSON(MANIFEST_URL, "Model manifest");
    const result = await prepareModelCache(manifest, message => {
      cacheStatus.textContent = message;
      console.log("[Model cache] " + message);
    }, (value, path) => {
      cacheStatus.textContent = "Caching " + Math.round(value) + "% · " + path;
    }, () => {});
    cacheStatus.textContent = "Ready · " + result.paths.length + " files · " +
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
    const manifest = await fetchJSON(MANIFEST_URL, "Model manifest");
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
