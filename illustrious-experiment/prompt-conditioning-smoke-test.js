import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";
import { acquireHeavyModelTest, releaseHeavyModelTest } from "./heavy-model-test-lock.js";
import { CLIPTokenizer } from "../vendor/transformers/transformers.min.js";

const MODEL_ID = "aaaaaeeeee/blip-sdxl-anime-wai-illustrious17";
const MODEL_BASE = "https://huggingface.co/" + MODEL_ID + "/resolve/main/";
const MANIFEST_URL = MODEL_BASE + "manifest.json";

ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

// The pack provides the original CLIP BPE assets, not tokenizer.json or
// tokenizer_config.json. Build the equivalent Transformers.js CLIP tokenizer
// directly from its vocab, merges, and special-token map.
const button = document.querySelector("#testPromptConditioning");
const status = document.querySelector("#promptConditioningStatus");
const results = document.querySelector("#promptConditioningSummary");
const promptInput = document.querySelector("#conditioningPrompt");
const negativePromptInput = document.querySelector("#conditioningNegativePrompt");

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
  results.append(row);
}

function finiteFp16(tensor, label, shape) {
  if (!tensor?.data || tensor.type !== "float16" || !tensor.dims) {
    throw new Error(label + " is not a readable FP16 tensor.");
  }
  if (tensor.dims.length !== shape.length ||
      tensor.dims.some((dimension, index) => dimension !== shape[index])) {
    throw new Error(label + " has shape [" + tensor.dims.join(", ") +
      "], expected [" + shape.join(", ") + "].");
  }
  if (!(tensor.data instanceof Float16Array) && !(tensor.data instanceof Uint16Array)) {
    throw new Error(label + " data is " + tensor.data.constructor.name +
      "; expected Float16Array or Uint16Array.");
  }

  const values = new Float16Array(tensor.data.buffer.slice(
    tensor.data.byteOffset,
    tensor.data.byteOffset + tensor.data.byteLength
  ));
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new Error(label + " contains a non-finite value at element " + i + ".");
    }
  }
  return values;
}

async function loadTokenizer(subfolder) {
  const folderURL = MODEL_BASE + subfolder + "/";

  async function readJSON(filename) {
    const url = folderURL + filename;
    const response = await fetch(url, { credentials: "omit" });
    if (!response.ok) {
      throw new Error("Tokenizer file request failed: " + filename +
        " · HTTP " + response.status + " · " + url);
    }
    try {
      return await response.json();
    } catch {
      throw new Error("Tokenizer file is not valid JSON: " + url);
    }
  }

  async function readText(filename) {
    const url = folderURL + filename;
    const response = await fetch(url, { credentials: "omit" });
    if (!response.ok) {
      throw new Error("Tokenizer file request failed: " + filename +
        " · HTTP " + response.status + " · " + url);
    }
    return response.text();
  }

  const [vocab, mergesText, specialMap] = await Promise.all([
    readJSON("vocab.json"),
    readText("merges.txt"),
    readJSON("special_tokens_map.json")
  ]);
  if (!vocab || typeof vocab !== "object" || Array.isArray(vocab)) {
    throw new Error(subfolder + "/vocab.json is not a token-to-ID object.");
  }

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
    throw new Error(subfolder + " CLIP vocabulary has unexpected BOS/EOS IDs: " +
      String(bosId) + " / " + String(eosId) + ".");
  }
  if (!Number.isSafeInteger(padId) || !Number.isSafeInteger(vocab[unkToken])) {
    throw new Error(subfolder + " CLIP vocabulary is missing its configured pad/unknown token.");
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
      if (pair.length !== 2) {
        throw new Error(subfolder + "/merges.txt has an invalid pair on line " + (index + 2) + ".");
      }
      return pair;
    });
  if (!merges.length) throw new Error(subfolder + "/merges.txt contains no BPE merges.");

  const normalizer = {
    type: "Sequence",
    normalizers: [
      { type: "NFC" },
      { type: "Replace", pattern: { Regex: "\\s+" }, content: " " },
      { type: "Lowercase" }
    ]
  };
  const preTokenizer = {
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
  };
  const tokenizerJSON = {
    version: "1.0",
    truncation: null,
    padding: null,
    added_tokens: [
      makeAddedToken(specialMap.bos_token, bosId, bosToken),
      makeAddedToken(specialMap.eos_token, eosId, eosToken)
    ],
    normalizer,
    pre_tokenizer: preTokenizer,
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

async function loadOnnxComponent(component, label) {
  if (!component?.onnx || !Array.isArray(component.shards) || !component.shards.length) {
    throw new Error("The manifest is missing " + label + " or its external-data shards.");
  }
  const externalData = component.shards.map(path => ({
    path: path.slice(path.lastIndexOf("/") + 1),
    data: MODEL_BASE + path
  }));
  return ort.InferenceSession.create(MODEL_BASE + component.onnx, {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "basic",
    externalData
  });
}

async function waitFrame() {
  return new Promise(resolve => requestAnimationFrame(() =>
    requestAnimationFrame(resolve)
  ));
}

button.addEventListener("click", async () => {
  button.disabled = true;
  results.replaceChildren();
  results.hidden = false;

  let tokenizerL;
  let tokenizerG;
  let sessionL;
  let sessionG;

  let lockAcquired = false;
  try {
    acquireHeavyModelTest("dual-CLIP prompt conditioning");
    lockAcquired = true;
    const prompt = promptInput.value.trim();
    const negativePrompt = negativePromptInput.value.trim();
    if (!prompt) throw new Error("Enter a positive prompt before testing text conditioning.");
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("The browser did not return a WebGPU adapter.");

    setStatus("Fetching the model manifest…");
    const response = await fetch(MANIFEST_URL, { credentials: "omit" });
    if (!response.ok) throw new Error("Model manifest request failed: HTTP " + response.status);
    const manifest = await response.json();

    addRow("Model pack", manifest.packageId || MODEL_ID);
    addRow("Positive prompt", prompt);
    addRow("Negative prompt", negativePrompt || "(empty)");
    addRow("CLIP-L component", ((manifest.components?.text_encoder?.bytes || 0) / 1024 / 1024).toFixed(1) + " MiB");
    addRow("CLIP-bigG component", ((manifest.components?.text_encoder_2?.bytes || 0) / 1024 / 1024).toFixed(1) + " MiB");
    addRow("Execution provider", "WebGPU · one text encoder session at a time");

    setStatus("Loading CLIP-L tokenizer from vocab.json, merges.txt, and special_tokens_map.json…");
    const started = performance.now();
    tokenizerL = await loadTokenizer("tokenizer");
    const encodeOptions = {
      padding: "max_length",
      truncation: true,
      max_length: 77
    };
    const encodedL = await tokenizerL(prompt, encodeOptions);
    const negativeEncodedL = await tokenizerL(negativePrompt, encodeOptions);

    setStatus("Loading CLIP-bigG tokenizer from vocab.json, merges.txt, and special_tokens_map.json…");
    tokenizerG = await loadTokenizer("tokenizer_2");
    const encodedG = await tokenizerG(prompt, encodeOptions);
    const negativeEncodedG = await tokenizerG(negativePrompt, encodeOptions);

    function makeInt64Ids(encoded, label) {
      const tensor = encoded?.input_ids;
      if (!tensor?.data || !Array.isArray(tensor.dims) ||
          tensor.dims.length !== 2 || tensor.dims[0] !== 1 || tensor.dims[1] !== 77) {
        throw new Error(label + " tokenizer did not produce input_ids [1, 77].");
      }
      const ids = new BigInt64Array(77);
      for (let i = 0; i < ids.length; i++) {
        const value = Number(tensor.data[i]);
        if (!Number.isSafeInteger(value)) {
          throw new Error(label + " tokenizer returned a non-integer token ID at position " + i + ".");
        }
        ids[i] = BigInt(value);
      }
      return ids;
    }

    const idsL = makeInt64Ids(encodedL, "CLIP-L positive");
    const negativeIdsL = makeInt64Ids(negativeEncodedL, "CLIP-L negative");
    const idsG = makeInt64Ids(encodedG, "CLIP-bigG positive");
    const negativeIdsG = makeInt64Ids(negativeEncodedG, "CLIP-bigG negative");
    const countTokens = encoded => {
      const mask = encoded.attention_mask?.data;
      return mask ? Array.from(mask).reduce((sum, value) => sum + (Number(value) > 0 ? 1 : 0), 0) : "unavailable";
    };
    addRow("CLIP-L tokenizer", "positive/negative int64 [1, 77] · " +
      countTokens(encodedL) + " / " + countTokens(negativeEncodedL) + " attended positions");
    addRow("CLIP-bigG tokenizer", "positive/negative int64 [1, 77] · " +
      countTokens(encodedG) + " / " + countTokens(negativeEncodedG) + " attended positions");

    // Stage 1: CLIP-L. Keep its small FP16 hidden-state output on CPU.
    setStatus("Loading CLIP-L and encoding the actual prompt…");
    const loadL = performance.now();
    sessionL = await loadOnnxComponent(manifest.components?.text_encoder, "CLIP-L");
    if (!sessionL.inputNames.includes("input_ids")) {
      throw new Error("CLIP-L graph is missing input_ids.");
    }
    let outputL = await sessionL.run({
      input_ids: new ort.Tensor("int64", idsL, [1, 77])
    });
    const hiddenL = finiteFp16(outputL.penultimate_hidden_state,
      "CLIP-L positive penultimate_hidden_state", [1, 77, 768]);
    outputL = null;
    let negativeOutputL = await sessionL.run({
      input_ids: new ort.Tensor("int64", negativeIdsL, [1, 77])
    });
    const negativeHiddenL = finiteFp16(negativeOutputL.penultimate_hidden_state,
      "CLIP-L negative penultimate_hidden_state", [1, 77, 768]);
    negativeOutputL = null;
    addRow("CLIP-L inference", ((performance.now() - loadL) / 1000).toFixed(1) + " s · positive + negative FP16 [1, 77, 768]");
    await sessionL.release();
    sessionL = null;
    await waitFrame();

    // Stage 2: CLIP-bigG. Keep penultimate states and pooled text embedding.
    setStatus("Loading CLIP-bigG and encoding the same prompt…");
    const loadG = performance.now();
    sessionG = await loadOnnxComponent(manifest.components?.text_encoder_2, "CLIP-bigG");
    if (!sessionG.inputNames.includes("input_ids")) {
      throw new Error("CLIP-bigG graph is missing input_ids.");
    }
    let outputG = await sessionG.run({
      input_ids: new ort.Tensor("int64", idsG, [1, 77])
    });
    const hiddenG = finiteFp16(outputG.penultimate_hidden_state,
      "CLIP-bigG positive penultimate_hidden_state", [1, 77, 1280]);
    const pooledG = finiteFp16(outputG.text_embeds, "CLIP-bigG positive text_embeds", [1, 1280]);
    outputG = null;
    let negativeOutputG = await sessionG.run({
      input_ids: new ort.Tensor("int64", negativeIdsG, [1, 77])
    });
    const negativeHiddenG = finiteFp16(negativeOutputG.penultimate_hidden_state,
      "CLIP-bigG negative penultimate_hidden_state", [1, 77, 1280]);
    const negativePooledG = finiteFp16(negativeOutputG.text_embeds,
      "CLIP-bigG negative text_embeds", [1, 1280]);
    negativeOutputG = null;
    addRow("CLIP-bigG inference", ((performance.now() - loadG) / 1000).toFixed(1) + " s · positive + negative FP16 outputs");
    await sessionG.release();
    sessionG = null;
    await waitFrame();

    // SDXL combines CLIP-L and CLIP-bigG penultimate hidden states per token.
    function combineHidden(hiddenOne, hiddenTwo, label) {
      const combined = new Float16Array(77 * 2048);
      for (let token = 0; token < 77; token++) {
        combined.set(hiddenOne.subarray(token * 768, token * 768 + 768), token * 2048);
        combined.set(hiddenTwo.subarray(token * 1280, token * 1280 + 1280), token * 2048 + 768);
      }
      for (let i = 0; i < combined.length; i++) {
        if (!Number.isFinite(combined[i])) throw new Error(label + " contains a non-finite value.");
      }
      return combined;
    }
    const positiveConditioning = combineHidden(hiddenL, hiddenG, "Positive SDXL conditioning");
    const negativeConditioning = combineHidden(negativeHiddenL, negativeHiddenG, "Negative SDXL conditioning");

    function checkFinite(values, label) {
      for (let i = 0; i < values.length; i++) {
        if (!Number.isFinite(values[i])) throw new Error(label + " contains a non-finite value at " + i + ".");
      }
    }
    checkFinite(pooledG, "Positive pooled text embedding");
    checkFinite(negativePooledG, "Negative pooled text embedding");

    addRow("Positive SDXL conditioning", "FP16 [1, 77, 2048] · all " + positiveConditioning.length.toLocaleString() + " values finite");
    addRow("Negative SDXL conditioning", "FP16 [1, 77, 2048] · all " + negativeConditioning.length.toLocaleString() + " values finite");
    addRow("Positive / negative pooled", "FP16 [1, 1280] / [1, 1280] · all finite");
    addRow("Total tokenization + load + inference", ((performance.now() - started) / 1000).toFixed(1) + " s");

    setStatus("Positive and negative prompts passed both tokenizers and CLIP encoders. The test assembled both SDXL conditioning pairs: encoder_hidden_states [1, 77, 2048] and text_embeds [1, 1280]. No UNet or VAE was loaded and no image was generated.", "success");
  } catch (error) {
    console.error("[Illustrious] Prompt conditioning smoke test failed:", error);
    setStatus("Prompt conditioning test failed: " + (error?.message || String(error)), "error");
  } finally {
    if (sessionL) {
      try { await sessionL.release(); } catch (error) {
        console.warn("[Illustrious] Could not release CLIP-L session:", error);
      }
    }
    if (sessionG) {
      try { await sessionG.release(); } catch (error) {
        console.warn("[Illustrious] Could not release CLIP-bigG session:", error);
      }
    }
    if (lockAcquired) releaseHeavyModelTest("dual-CLIP prompt conditioning");
    button.disabled = false;
  }
});
