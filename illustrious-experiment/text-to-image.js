import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";
import { CLIPTokenizer } from "../vendor/transformers/transformers.min.js";
import { acquireHeavyModelTest, releaseHeavyModelTest } from "./heavy-model-test-lock.js";
import { preparePersonalAmiComponent } from "./personal-ami-lora.js";

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

let cancelRequested = false;
let generationStarted = 0;
let imageObjectUrl = null;

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
  const response = await fetch(url, { credentials: "omit" });
  if (!response.ok) throw new Error(label + " request failed: HTTP " + response.status);
  return response.json();
}

async function loadTokenizer(subfolder) {
  const folderURL = MODEL_BASE + subfolder + "/";

  async function readJSON(filename) {
    const url = folderURL + filename;
    const response = await fetch(url, { credentials: "omit" });
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

async function loadSession(manifest, key, label, patchedExternalData = null) {
  const component = manifest?.components?.[key];
  const started = performance.now();
  log("Loading " + label + " (" + formatBytes(component?.bytes) + ")…");
  const session = await ort.InferenceSession.create(MODEL_BASE + component.onnx, {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "basic",
    externalData: patchedExternalData || externalData(component)
  });
  log(label + " session created in " + ((performance.now() - started) / 1000).toFixed(2) +
    "s · inputs: " + session.inputNames.join(", "));
  return session;
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
    const positive = await sessionL.run({ input_ids: new ort.Tensor("int64", idsL, [1, 77]) });
    hiddenL = fp16Values(positive.penultimate_hidden_state, "CLIP-L positive hidden", [1, 77, 768]);
    const negative = await sessionL.run({ input_ids: new ort.Tensor("int64", negativeIdsL, [1, 77]) });
    negativeHiddenL = fp16Values(negative.penultimate_hidden_state, "CLIP-L negative hidden", [1, 77, 768]);
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
    const positive = await sessionG.run({ input_ids: new ort.Tensor("int64", idsG, [1, 77]) });
    hiddenG = fp16Values(positive.penultimate_hidden_state, "CLIP-bigG positive hidden", [1, 77, 1280]);
    pooledG = fp16Values(positive.text_embeds, "CLIP-bigG positive pooled", [1, 1280]);
    const negative = await sessionG.run({ input_ids: new ort.Tensor("int64", negativeIdsG, [1, 77]) });
    negativeHiddenG = fp16Values(negative.penultimate_hidden_state, "CLIP-bigG negative hidden", [1, 77, 1280]);
    negativePooledG = fp16Values(negative.text_embeds, "CLIP-bigG negative pooled", [1, 1280]);
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

function makeBatchSample(scaled, latentHeight, latentWidth) {
  const oneSize = 4 * latentHeight * latentWidth;
  const batch = new Float16Array(oneSize * 2);
  for (let i = 0; i < oneSize; i++) {
    batch[i] = scaled[i];
    batch[oneSize + i] = scaled[i];
  }
  return batch;
}

function makeBatchTimeIds(height, width) {
  const one = [height, width, 0, 0, height, width];
  const batch = new Float32Array(12);
  batch.set(one, 0);
  batch.set(one, 6);
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

async function runUnetStep(sessions, conditioning, scaledLatents, timestep, width, height, cfg, noiseStep) {
  const latentWidth = width / 8;
  const latentHeight = height / 8;
  const batchSample = makeBatchSample(scaledLatents, latentHeight, latentWidth);
  const batch = 2;
  const shape = (channels, h, w) => [batch, channels, h, w];
  const timings = {};
  let outputsA;
  let outputsB;
  let outputsC;

  const startedA = performance.now();
  outputsA = await sessions.a.run({
    sample: new ort.Tensor("float16", batchSample, [batch, 4, latentHeight, latentWidth]),
    timestep: new ort.Tensor("float32", new Float32Array([timestep, timestep]), [batch]),
    encoder_hidden_states: new ort.Tensor("float16", conditioning.encoderHidden, [batch, 77, 2048]),
    text_embeds: new ort.Tensor("float16", conditioning.textEmbeds, [batch, 1280]),
    time_ids: new ort.Tensor("float32", makeBatchTimeIds(height, width), [batch, 6])
  });
  timings.a = (performance.now() - startedA) / 1000;
  checkCancelled();

  const aHidden = fp16Values(outputsA.hidden, "UNET_A hidden", shape(1280, latentHeight / 4, latentWidth / 4));
  const aEmb = fp16Values(outputsA.emb, "UNET_A emb", [batch, 1280]);
  const skips = {};
  for (const index of [0, 1, 2]) {
    skips[index] = fp16Values(outputsA["skip_" + index], "UNET_A skip_" + index,
      shape(320, latentHeight, latentWidth));
  }
  skips[3] = fp16Values(outputsA.skip_3, "UNET_A skip_3",
    shape(320, latentHeight / 2, latentWidth / 2));
  skips[4] = fp16Values(outputsA.skip_4, "UNET_A skip_4",
    shape(640, latentHeight / 2, latentWidth / 2));
  skips[5] = fp16Values(outputsA.skip_5, "UNET_A skip_5",
    shape(640, latentHeight / 2, latentWidth / 2));
  skips[6] = fp16Values(outputsA.skip_6, "UNET_A skip_6",
    shape(640, latentHeight / 4, latentWidth / 4));
  skips[7] = fp16Values(outputsA.skip_7, "UNET_A skip_7",
    shape(1280, latentHeight / 4, latentWidth / 4));
  skips[8] = fp16Values(outputsA.skip_8, "UNET_A skip_8",
    shape(1280, latentHeight / 4, latentWidth / 4));
  outputsA = null;

  const startedB = performance.now();
  outputsB = await sessions.b.run({
    hidden: new ort.Tensor("float16", aHidden, shape(1280, latentHeight / 4, latentWidth / 4)),
    emb: new ort.Tensor("float16", aEmb, [batch, 1280]),
    encoder_hidden_states: new ort.Tensor("float16", conditioning.encoderHidden, [batch, 77, 2048]),
    skip_6: new ort.Tensor("float16", skips[6], shape(640, latentHeight / 4, latentWidth / 4)),
    skip_7: new ort.Tensor("float16", skips[7], shape(1280, latentHeight / 4, latentWidth / 4)),
    skip_8: new ort.Tensor("float16", skips[8], shape(1280, latentHeight / 4, latentWidth / 4))
  });
  timings.b = (performance.now() - startedB) / 1000;
  checkCancelled();
  const hiddenOut = fp16Values(outputsB.hidden_out, "UNET_B hidden_out",
    shape(1280, latentHeight / 2, latentWidth / 2));
  outputsB = null;

  const startedC = performance.now();
  outputsC = await sessions.c.run({
    hidden: new ort.Tensor("float16", hiddenOut, shape(1280, latentHeight / 2, latentWidth / 2)),
    emb: new ort.Tensor("float16", aEmb, [batch, 1280]),
    encoder_hidden_states: new ort.Tensor("float16", conditioning.encoderHidden, [batch, 77, 2048]),
    skip_0: new ort.Tensor("float16", skips[0], shape(320, latentHeight, latentWidth)),
    skip_1: new ort.Tensor("float16", skips[1], shape(320, latentHeight, latentWidth)),
    skip_2: new ort.Tensor("float16", skips[2], shape(320, latentHeight, latentWidth)),
    skip_3: new ort.Tensor("float16", skips[3], shape(320, latentHeight / 2, latentWidth / 2)),
    skip_4: new ort.Tensor("float16", skips[4], shape(640, latentHeight / 2, latentWidth / 2)),
    skip_5: new ort.Tensor("float16", skips[5], shape(640, latentHeight / 2, latentWidth / 2))
  });
  timings.c = (performance.now() - startedC) / 1000;
  const noisePrediction = fp16Values(outputsC.out_sample, "UNET_C out_sample",
    [batch, 4, latentHeight, latentWidth]);
  outputsC = null;

  const latentSize = 4 * latentHeight * latentWidth;
  const guided = new Float32Array(latentSize);
  for (let i = 0; i < latentSize; i++) {
    const unconditional = noisePrediction[i];
    const conditional = noisePrediction[latentSize + i];
    guided[i] = unconditional + cfg * (conditional - unconditional);
  }

  log("Step denoiser timing · A " + timings.a.toFixed(2) + "s · B " +
    timings.b.toFixed(2) + "s · C " + timings.c.toFixed(2) + "s · CFG " + cfg);
  return { guided, timings };
}

async function releaseSessions(sessions) {
  for (const key of ["c", "b", "a"]) {
    const session = sessions[key];
    if (!session) continue;
    try {
      await session.release();
      sessions[key] = null;
      await yieldFrames();
      log("Released UNET_" + key.toUpperCase() + ".");
    } catch (error) {
      log("Warning: failed to release UNET_" + key.toUpperCase() + ": " + (error?.message || error));
    }
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

async function generate(options) {
  const manifest = await fetchJSON(MANIFEST_URL, "Model manifest");
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
  log("CLIP encoders run sequentially; all three UNet sessions stay resident through denoising; VAE loads after UNet release.");
  if (options.personalAmi) {
    log("Personal Ami enabled · " + options.personalAmi.file.name + " · strength " + options.personalAmi.strength.toFixed(3) + " · all 980 mapped pairs will be patched.");
  } else {
    log("Personal Ami disabled · using unmodified WAI Illustrious v17 base weights.");
  }

  checkCancelled();
  setProgress(2, "Encoding prompts");
  setStatus("Encoding positive and negative prompts…");
  const conditioning = await encodePrompts(manifest, options.prompt, options.negative, options.personalAmi);
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
  const sessions = { a: null, b: null, c: null };
  let totalUnetExecutionSeconds = 0;
  try {
    setProgress(8, "Loading UNet partitions");
    setStatus("Loading UNET_A, UNET_B, and UNET_C together. This requires several GiB of memory…");
    const patchedA = options.personalAmi
      ? await preparePersonalAmiComponent(options.personalAmi.file, manifest, "unet_a", options.personalAmi.strength, log, checkCancelled)
      : null;
    sessions.a = await loadSession(manifest, "unet_a", "UNET_A", patchedA?.externalData);
    checkCancelled();
    const patchedB = options.personalAmi
      ? await preparePersonalAmiComponent(options.personalAmi.file, manifest, "unet_b", options.personalAmi.strength, log, checkCancelled)
      : null;
    sessions.b = await loadSession(manifest, "unet_b", "UNET_B", patchedB?.externalData);
    checkCancelled();
    const patchedC = options.personalAmi
      ? await preparePersonalAmiComponent(options.personalAmi.file, manifest, "unet_c", options.personalAmi.strength, log, checkCancelled)
      : null;
    sessions.c = await loadSession(manifest, "unet_c", "UNET_C", patchedC?.externalData);
    validateSessionInputs(sessions.a, ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"], "UNET_A");
    validateSessionInputs(sessions.b, ["hidden", "emb", "encoder_hidden_states", "skip_6", "skip_7", "skip_8"], "UNET_B");
    validateSessionInputs(sessions.c, ["hidden", "emb", "encoder_hidden_states", "skip_0", "skip_1", "skip_2", "skip_3", "skip_4", "skip_5"], "UNET_C");
    log("All three UNet sessions are resident. Beginning " + options.steps + " denoising steps.");
    await yieldFrames();

    for (let step = 0; step < options.steps; step++) {
      checkCancelled();
      const stepStarted = performance.now();
      const scaled = scheduler.scaleModelInput(latents, step);
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
      const epsilon = denoiser.guided;
      const noise = createNormalNoise(latentSize, options.seed + step + 1);
      latents = scheduler.step(epsilon, step, latents, noise);
      for (let i = 0; i < latents.length; i++) {
        if (!Number.isFinite(latents[i])) throw new Error("Denoising produced a non-finite latent at " + i + ".");
      }
      const percent = 8 + ((step + 1) / options.steps) * 82;
      setProgress(percent, "Denoising " + (step + 1) + " / " + options.steps);
      setStatus("Denoising step " + (step + 1) + " of " + options.steps + "…");
      log("Step " + (step + 1) + "/" + options.steps + " scheduler update " +
        ((performance.now() - stepStarted) / 1000).toFixed(2) + "s · sigma " +
        scheduler.sigmas[step].toFixed(5) + " → " + scheduler.sigmas[step + 1].toFixed(5));
      await yieldFrames();
    }
  } finally {
    await releaseSessions(sessions);
  }
  log("Accumulated UNet execution time " + totalUnetExecutionSeconds.toFixed(2) +
    "s across " + options.steps + " steps (sum of A + B + C inference calls).");

  checkCancelled();
  log("Denoising complete. Loading VAE decoder after releasing all UNet partitions.");
  setProgress(92, "Loading VAE decoder");
  setStatus("Loading VAE decoder…");
  let vaeSession;
  try {
    vaeSession = await loadSession(manifest, "vae_decoder", "VAE decoder");
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
  } finally {
    if (vaeSession) {
      try {
        await vaeSession.release();
        await yieldFrames();
        log("Released VAE decoder.");
      } catch (error) {
        log("Warning: could not release VAE decoder: " + (error?.message || error));
      }
    }
  }
}

button.addEventListener("click", async () => {
  let options;
  try {
    options = readOptions();
  } catch (error) {
    setStatus(error.message || String(error), "error");
    return;
  }

  button.disabled = true;
  cancelButton.disabled = false;
  cancelRequested = false;
  logOutput.textContent = "";
  progress.value = 0;
  setProgress(0, "Starting");
  generationStarted = performance.now();
  let lockAcquired = false;

  try {
    acquireHeavyModelTest("full Illustrious image generation");
    lockAcquired = true;
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
