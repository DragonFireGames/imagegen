import * as ort from "./vendor/ort.jspi.bundle.min.mjs";
import { PreTrainedTokenizer, env as transformersEnv } from "./vendor/transformers/transformers.min.js";
import { clearModelCache, getCachedModelFile, loadModelPack } from "./model-store.js";

ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.logLevel = "warning";
transformersEnv.useBrowserCache = true;
transformersEnv.allowRemoteModels = false;
transformersEnv.allowLocalModels = false;

let activePack = null;
let tokenizerPromise = null;
let operationController = null;

function post(type, fields = {}, transfer = []) {
  self.postMessage({ type, ...fields }, transfer);
}

function status(label, extra = "") {
  post("status", { label, extra });
}

function perf(label, started, details = "") {
  const elapsed = performance.now() - started;
  post("status", {
    label: "PERF · " + label,
    extra: elapsed.toFixed(1) + " ms" + (details ? " · " + details : ""),
  });
  return elapsed;
}

function perfMark(label, details = "") {
  post("status", { label: "PERF · " + label, extra: details });
}

function toBigIntIds(tokenizer, prompt) {
  return Promise.resolve(tokenizer._call(prompt, {
    padding: "max_length",
    truncation: true,
    max_length: 77,
  })).then((encoded) => {
    const values = encoded.input_ids?.data || encoded.input_ids;
    if (!values || values.length < 77) throw new Error("Tokenizer returned fewer than 77 token IDs.");
    return BigInt64Array.from(Array.from(values).slice(0, 77), (value) => BigInt(Number(value)));
  });
}

async function loadTokenizers() {
  if (!tokenizerPromise) {
    const loadOne = async (folder, label) => {
      const tokenizerStarted = performance.now();
      const loadJson = async (filename) => {
        const url = new URL(folder + "/" + filename, activePack.baseUrl);
        status("Downloading " + label, filename);
        const fetchStarted = performance.now();
        const response = await fetch(url);
        perf(label + " fetch " + filename, fetchStarted, response.status + " · " + (response.headers.get("content-length") || "unknown") + " bytes");
        if (!response.ok) {
          throw new Error("Could not load bundled " + label + " asset " + filename +
            " (HTTP " + response.status + "): " + url);
        }
        try {
          return await response.json();
        } catch (error) {
          throw new Error("Invalid JSON in bundled tokenizer asset " + url + ": " + error.message);
        }
      };

      const [tokenizerJSON, tokenizerConfig] = await Promise.all([
        loadJson("tokenizer.json"),
        loadJson("tokenizer_config.json"),
      ]);
      if (!tokenizerJSON || typeof tokenizerJSON !== "object" || !tokenizerJSON.model) {
        throw new Error("Bundled tokenizer " + folder + "/tokenizer.json has no tokenizer model.");
      }
      const tokenizer = new PreTrainedTokenizer(tokenizerJSON, tokenizerConfig);
      perf(label + " construction", tokenizerStarted, "vocab " + Object.keys(tokenizerJSON.model.vocab || {}).length + " entries");
      return tokenizer;
    };

    tokenizerPromise = Promise.all([
      loadOne("tokenizer", "tokenizer 1"),
      loadOne("tokenizer_2", "tokenizer 2"),
    ]).catch((error) => {
      tokenizerPromise = null;
      throw error;
    });
  }
  return tokenizerPromise;
}

async function createSession(componentName, provider) {
  const sessionStarted = performance.now();
  const component = activePack.manifest.components[componentName];
  if (!component) throw new Error("The model manifest is missing component: " + componentName);
  const packVersion = activePack.manifest.packId || activePack.manifest.sourceRevision || "unversioned";
  const graphStarted = performance.now();
  const graphFile = await getCachedModelFile(activePack.baseUrl, component.graphBytesAsset, packVersion);
  perf(componentName + " graph cache/read", graphStarted, (graphFile.size / 1024 ** 2).toFixed(1) + " MiB");
  const externalData = [];
  for (const shard of component.externalData || []) {
    const shardStarted = performance.now();
    const data = await getCachedModelFile(activePack.baseUrl, shard.asset, packVersion);
    perf(componentName + " shard " + shard.asset, shardStarted, (data.size / 1024 ** 2).toFixed(1) + " MiB");
    externalData.push({ path: shard.path, data });
  }
  status("Preparing " + componentName.replaceAll("_", " "), provider === "webgpu" ? "WebGPU" : "WebAssembly");
  const session = await ort.InferenceSession.create(await graphFile.arrayBuffer(), {
    executionProviders: [provider],
    graphOptimizationLevel: componentName.startsWith("text_encoder") ? "basic" : "all",
    externalData,
  });
  perf(componentName + " session initialization", sessionStarted, provider);
  return session;
}

function asFloat32(tensor) {
  if (tensor.type === "float16") {
    const data = tensor.data;
    if (typeof Float16Array !== "undefined" && data instanceof Float16Array) return Float32Array.from(data);
    if (data instanceof Uint16Array) {
      const output = new Float32Array(data.length);
      for (let i = 0; i < output.length; i++) output[i] = halfToFloat(data[i]);
      return output;
    }
    throw new Error("Unsupported float16 output buffer: " + Object.prototype.toString.call(data));
  }
  return Float32Array.from(tensor.data);
}

function floatToHalf(value) {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = value;
  const bits = u32[0];
  const sign = (bits >>> 16) & 0x8000;
  let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >> (1 - exponent);
    return sign | ((mantissa + 0x1000 + ((mantissa >> 13) & 1)) >> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  mantissa += 0x1000 + ((mantissa >> 13) & 1);
  if (mantissa & 0x800000) {
    mantissa = 0;
    exponent++;
    if (exponent >= 31) return sign | 0x7c00;
  }
  return sign | (exponent << 10) | (mantissa >> 13);
}

function halfToFloat(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 31;
  const fraction = value & 1023;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function tensorF16(values, dims) {
  // ONNX Runtime's newer browser build accepts numeric Float16Array values.
  // Older runtimes use Uint16Array as the raw IEEE-754 binary16 bit container.
  if (typeof Float16Array !== "undefined") {
    return new ort.Tensor("float16", Float16Array.from(values), dims);
  }
  return new ort.Tensor("float16", Uint16Array.from(values, floatToHalf), dims);
}

async function encodeWithSession(session, tokenizer, prompt, withPooled = false) {
  const tokenizeStarted = performance.now();
  const ids = await toBigIntIds(tokenizer, prompt);
  perf("tokenization", tokenizeStarted, prompt.length + " prompt characters");
  const encodeStarted = performance.now();
  const outputs = await session.run({
    input_ids: new ort.Tensor("int64", ids, [1, 77]),
  });
  perf("text encoder inference", encodeStarted, withPooled ? "hidden + pooled outputs" : "hidden output");
  const result = { hidden: asFloat32(outputs.hidden_states) };
  if (withPooled) result.pooled = asFloat32(outputs.text_embeds);
  return result;
}

function combineTextEmbeddings(hidden1, hidden2, pooled) {
  if (hidden1.length !== 77 * 768 || hidden2.length !== 77 * 1280 || pooled.length !== 1280) {
    throw new Error("Text encoder output shape mismatch. Expected SDXL penultimate states and pooled CLIP-bigG embedding.");
  }
  const hidden = new Float32Array(77 * 2048);
  for (let token = 0; token < 77; token++) {
    hidden.set(hidden1.subarray(token * 768, token * 768 + 768), token * 2048);
    hidden.set(hidden2.subarray(token * 1280, token * 1280 + 1280), token * 2048 + 768);
  }
  return { hidden, pooled };
}

async function encodePrompts(prompt, negativePrompt, tokenizers, cfgScale) {
  const promptStarted = performance.now();
  const runNegative = Boolean(negativePrompt.trim()) && cfgScale > 1;
  perfMark("prompt encoding started", "negative prompt " + (runNegative ? "enabled" : "disabled"));
  let positive1, negative1, positive2, negative2;
  status("Loading CLIP-L encoder", "WebAssembly · 246 MB source weights");
  const encoder1 = await createSession("text_encoder", "wasm");
  try {
    status("Encoding positive prompt · CLIP-L");
    positive1 = await encodeWithSession(encoder1, tokenizers[0], prompt);
    if (runNegative) {
      status("Encoding negative prompt · CLIP-L");
      negative1 = await encodeWithSession(encoder1, tokenizers[0], negativePrompt);
    }
  } finally {
    await encoder1.release();
  }

  status("Loading CLIP-bigG encoder", "WebAssembly · 1.4 GB source weights");
  const encoder2 = await createSession("text_encoder_2", "wasm");
  try {
    status("Encoding positive prompt · CLIP-bigG");
    positive2 = await encodeWithSession(encoder2, tokenizers[1], prompt, true);
    if (runNegative) {
      status("Encoding negative prompt · CLIP-bigG");
      negative2 = await encodeWithSession(encoder2, tokenizers[1], negativePrompt, true);
    }
  } finally {
    await encoder2.release();
  }

  const positive = combineTextEmbeddings(positive1.hidden, positive2.hidden, positive2.pooled);
  const negative = runNegative
    ? combineTextEmbeddings(negative1.hidden, negative2.hidden, negative2.pooled)
    : { hidden: new Float32Array(77 * 2048), pooled: new Float32Array(1280) };
  perf("both CLIP encoders total", promptStarted);
  return { positive, negative };
}

function makeRng(seed) {
  let state = (seed >>> 0) || 0x6d2b79f5;
  function random() {
    state = (state + 0x6d2b79f5) | 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  }
  return () => {
    const u1 = Math.max(random(), 1e-12);
    const u2 = random();
    const radius = Math.sqrt(-2 * Math.log(u1));
    return [radius * Math.cos(2 * Math.PI * u2), radius * Math.sin(2 * Math.PI * u2)];
  };
}

function makeNoise(length, randomNormal) {
  const values = new Float32Array(length);
  for (let i = 0; i < length; i += 2) {
    const pair = randomNormal();
    values[i] = pair[0];
    if (i + 1 < length) values[i + 1] = pair[1];
  }
  return values;
}

function makeSigmas(config, steps) {
  if (config.prediction_type && config.prediction_type !== "epsilon") {
    throw new Error("This browser sampler currently supports epsilon prediction only; model config uses " + config.prediction_type + ".");
  }
  if (config.use_karras_sigmas || config.use_exponential_sigmas || config.use_beta_sigmas || config.rescale_betas_zero_snr) {
    throw new Error("This browser sampler does not yet implement the configured sigma schedule variant.");
  }

  const count = Number(config.num_train_timesteps || 1000);
  const start = Number(config.beta_start ?? 0.00085);
  const end = Number(config.beta_end ?? 0.012);
  const schedule = config.beta_schedule || "scaled_linear";
  const cumulative = new Float64Array(count);
  let product = 1;
  for (let i = 0; i < count; i++) {
    let beta;
    if (schedule === "scaled_linear") {
      const root = Math.sqrt(start) + (Math.sqrt(end) - Math.sqrt(start)) * (i / (count - 1));
      beta = root * root;
    } else if (schedule === "linear") {
      beta = start + (end - start) * (i / (count - 1));
    } else {
      throw new Error("Unsupported beta schedule: " + schedule);
    }
    product *= 1 - beta;
    cumulative[i] = product;
  }

  const trainSigmas = new Float32Array(count);
  for (let i = 0; i < count; i++) trainSigmas[i] = Math.sqrt((1 - cumulative[i]) / cumulative[i]);

  const timesteps = new Float32Array(steps);
  const sigmas = new Float32Array(steps + 1);
  const spacing = config.timestep_spacing || "leading";
  const offset = Number(config.steps_offset || 0);
  if (spacing === "leading") {
    const stepRatio = Math.floor(count / steps);
    for (let i = 0; i < steps; i++) {
      const t = Math.round((steps - 1 - i) * stepRatio) + offset;
      if (t < 0 || t >= count) throw new Error("Scheduler timestep is outside the training schedule.");
      timesteps[i] = t;
      sigmas[i] = trainSigmas[t];
    }
  } else if (spacing === "linspace") {
    for (let i = 0; i < steps; i++) {
      const t = Math.round((count - 1) * (1 - i / (steps - 1)));
      timesteps[i] = t;
      sigmas[i] = trainSigmas[t];
    }
  } else if (spacing === "trailing") {
    const stepRatio = count / steps;
    for (let i = 0; i < steps; i++) {
      const t = Math.round(count - i * stepRatio) - 1;
      timesteps[i] = t;
      sigmas[i] = trainSigmas[t];
    }
  } else {
    throw new Error("Unsupported timestep spacing: " + spacing);
  }
  sigmas[steps] = 0;
  // Diffusers EulerDiscreteScheduler scales the initial noise differently for
  // leading spacing than for linspace/trailing. NoobAI's config uses leading.
  const initNoiseSigma = spacing === "linspace" || spacing === "trailing"
    ? sigmas[0]
    : Math.sqrt(sigmas[0] * sigmas[0] + 1);
  return { sigmas, timesteps, initNoiseSigma };
}

async function predictNoise(session, latents, sigma, timestep, condition, height, width) {
  const latentHeight = height / 8;
  const latentWidth = width / 8;
  const scale = 1 / Math.sqrt(sigma * sigma + 1);
  const scaled = new Float32Array(latents.length);
  for (let i = 0; i < latents.length; i++) scaled[i] = latents[i] * scale;

  const timeIds = Float32Array.from([height, width, 0, 0, height, width]);
  const output = await session.run({
    sample: tensorF16(scaled, [1, 4, latentHeight, latentWidth]),
    timestep: tensorF16([timestep], [1]),
    encoder_hidden_states: tensorF16(condition.hidden, [1, 77, 2048]),
    text_embeds: tensorF16(condition.pooled, [1, 1280]),
    time_ids: tensorF16(timeIds, [1, 6]),
  });
  const noise = asFloat32(output.noise_pred);
  if (noise.length !== latents.length) {
    throw new Error("U-Net prediction length mismatch: expected " + latents.length + ", received " + noise.length + ".");
  }
  return noise;
}

async function generate(message) {
  if (!activePack) throw new Error("Load a NoobAI browser model pack first.");
  const { prompt, negativePrompt, width, height, steps, cfgScale, requestedSeed } = message;
  const sampler = message.sampler === "euler" ? "euler" : "euler_a";
  if (!prompt.trim()) throw new Error("Enter a positive prompt.");
  if (![width, height].every((n) => Number.isInteger(n) && n >= 512 && n <= 1216 && n % 64 === 0)) {
    throw new Error("Width and height must be 512–1216 and divisible by 64.");
  }
  if (!Number.isInteger(steps) || steps < 10 || steps > 50) throw new Error("Steps must be 10–50.");
  if (!Number.isFinite(cfgScale) || cfgScale < 1 || cfgScale > 12) throw new Error("CFG must be 1–12.");

  const started = performance.now();
  perfMark("generation started", width + " × " + height + " · " + steps + " steps · CFG " + cfgScale);
  const seed = requestedSeed < 0 ? Math.floor(Math.random() * 2147483647) : requestedSeed;
  perfMark("seed selected", String(seed));
  const tokenizerStarted = performance.now();
  const tokenizers = await loadTokenizers();
  perf("load both tokenizers", tokenizerStarted);
  const { positive, negative } = await encodePrompts(prompt, negativePrompt, tokenizers, cfgScale);

  const latentHeight = height / 8;
  const latentWidth = width / 8;
  const latentSize = 4 * latentHeight * latentWidth;
  perfMark("latent allocation", latentSize.toLocaleString() + " values · " + (latentSize * 4 / 1024 ** 2).toFixed(1) + " MiB FP32");
  const rng = makeRng(seed);
  const schedulerStarted = performance.now();
  const { sigmas, timesteps, initNoiseSigma } = makeSigmas(activePack.manifest.scheduler || {}, steps);
  perf("scheduler setup", schedulerStarted, "initial sigma " + initNoiseSigma.toFixed(4));
  let latents = makeNoise(latentSize, rng);
  for (let i = 0; i < latentSize; i++) latents[i] *= initNoiseSigma;

  status("Loading SDXL U-Net", "WebGPU");
  const unetStarted = performance.now();
  const unet = await createSession("unet", "webgpu");
  perfMark("denoising loop started", "U-Net ready; " + (cfgScale > 1 && negativePrompt.trim() ? "2 predictions per step" : "1 prediction per step"));
  try {
    for (let step = 0; step < steps; step++) {
      const stepStarted = performance.now();
      if (operationController?.signal.aborted) throw new Error("Generation cancelled.");
      const sigma = sigmas[step];
      const nextSigma = sigmas[step + 1];
      const conditionalStarted = performance.now();
      const conditional = await predictNoise(unet, latents, sigma, timesteps[step], positive, height, width);
      const conditionalMs = performance.now() - conditionalStarted;
      let unconditionalMs = 0;
      let prediction = conditional;
      const useCFG = cfgScale > 1 && Boolean(negativePrompt.trim());
      if (useCFG) {
        const unconditionalStarted = performance.now();
        const unconditional = await predictNoise(unet, latents, sigma, timesteps[step], negative, height, width);
        unconditionalMs = performance.now() - unconditionalStarted;
        prediction = new Float32Array(latentSize);
        for (let i = 0; i < latentSize; i++) {
          prediction[i] = unconditional[i] + cfgScale * (conditional[i] - unconditional[i]);
        }
      }

      if (sampler === "euler_a") {
        // Euler ancestral update uses the source scheduler's exact sigmas and
        // injects a fresh, seeded noise sample for the ancestral component.
        const sigmaSquared = sigma * sigma;
        const nextSquared = nextSigma * nextSigma;
        const sigmaUp = nextSigma === 0 || sigma === 0
          ? 0
          : Math.min(nextSigma, Math.sqrt(Math.max(0, nextSquared * (sigmaSquared - nextSquared) / sigmaSquared)));
        const sigmaDown = Math.sqrt(Math.max(0, nextSquared - sigmaUp * sigmaUp));
        const dt = sigmaDown - sigma;
        const noise = sigmaUp > 0 ? makeNoise(latentSize, rng) : null;
        for (let i = 0; i < latentSize; i++) {
          latents[i] += prediction[i] * dt;
          if (noise) latents[i] += noise[i] * sigmaUp;
        }
      } else {
        // Deterministic EulerDiscrete update: d(sample)/d(sigma) = epsilon.
        const dt = nextSigma - sigma;
        for (let i = 0; i < latentSize; i++) latents[i] += prediction[i] * dt;
      }
      const stepMs = performance.now() - stepStarted;
      if (step === 0 || (step + 1) % 5 === 0 || step + 1 === steps) {
        perfMark("U-Net step " + (step + 1) + "/" + steps,
          "total " + stepMs.toFixed(0) + " ms · conditional " + conditionalMs.toFixed(0) + " ms" +
          (useCFG ? " · unconditional " + unconditionalMs.toFixed(0) + " ms" : "") +
          " · sigma " + sigma.toFixed(4));
      }
      post("step", { step: step + 1, total: steps, seed });
    }
  } finally {
    await unet.release();
    perf("U-Net load + denoising + release", unetStarted);
  }

  status("Loading VAE decoder", "WebGPU");
  const vaeStarted = performance.now();
  const vae = await createSession("vae_decoder", "webgpu");
  let decoded;
  try {
    const scaledLatents = new Float32Array(latents.length);
    const vaeScale = Number(activePack.manifest.vaeScalingFactor || 0.13025);
    for (let i = 0; i < latents.length; i++) scaledLatents[i] = latents[i] / vaeScale;
    const decodeStarted = performance.now();
    const output = await vae.run({
      latent_sample: new ort.Tensor("float32", scaledLatents, [1, 4, latentHeight, latentWidth]),
    });
    perf("VAE decode inference", decodeStarted, width + " × " + height);
    decoded = output.sample;
  } finally {
    await vae.release();
    perf("VAE session load + decode + release", vaeStarted);
  }

  const image = asFloat32(decoded);
  const expected = height * width * 3;
  if (image.length !== expected) {
    throw new Error("VAE image output shape mismatch: expected " + expected + " values, got " + image.length + ".");
  }
  for (let i = 0; i < image.length; i++) {
    if (!Number.isFinite(image[i])) throw new Error("VAE output contains a non-finite value at index " + i + ".");
  }
  const outputBuffer = image.buffer.slice(image.byteOffset, image.byteOffset + image.byteLength);
  perf("total generation", started, width + " × " + height + " · seed " + seed);
  perfMark("output prepared", (outputBuffer.byteLength / 1024 ** 2).toFixed(1) + " MiB RGB float32");
  post("result", {
    buffer: outputBuffer,
    dims: [1, 3, height, width],
    seed,
    elapsed: (performance.now() - started) / 1000,
    sampler,
  }, [outputBuffer]);
}

self.onmessage = async (event) => {
  const message = event.data || {};
  try {
    if (message.op === "load") {
      operationController?.abort();
      operationController = new AbortController();
      tokenizerPromise = null;
      const baseUrl = message.baseUrl.trim().replace(/\/+$/, "");
      if (!baseUrl.startsWith("https://")) throw new Error("Use an HTTPS model-pack URL.");
      status("Reading NoobAI model manifest");
      const packLoadStarted = performance.now();
      let activeAsset = null;
      let activeAssetStarted = 0;
      let activeAssetCached = false;
      activePack = await loadModelPack(baseUrl, (progress) => {
        if (progress.file !== activeAsset) {
          if (activeAsset) {
            const elapsed = performance.now() - activeAssetStarted;
            perfMark("model asset " + (activeAssetCached ? "cache hit" : "downloaded") + " · " + activeAsset,
              elapsed.toFixed(0) + " ms");
          }
          activeAsset = progress.file;
          activeAssetStarted = performance.now();
          activeAssetCached = Boolean(progress.cached);
          perfMark("model asset " + (activeAssetCached ? "checking cache" : "fetching") + " · " + activeAsset);
        }
        const percent = progress.total ? Math.round(progress.loaded / progress.total * 100) : 0;
        post("progress", {
          label: progress.phase === "cache" ? "Checking browser cache" : "Downloading " + progress.file,
          loaded: progress.loaded,
          total: progress.total,
          percent,
        });
      }, operationController.signal);
      if (activeAsset) {
        perfMark("model asset " + (activeAssetCached ? "cache hit" : "downloaded") + " · " + activeAsset,
          (performance.now() - activeAssetStarted).toFixed(0) + " ms");
      }
      perf("model pack load/cache", packLoadStarted, (activePack.cachedBytes / 1024 ** 3).toFixed(2) + " GiB cached");
      post("model-ready", {
        model: activePack.manifest.model,
        totalBytes: activePack.cachedBytes,
      });
    } else if (message.op === "generate") {
      operationController?.abort();
      operationController = new AbortController();
      await generate(message);
    } else if (message.op === "clear-cache") {
      operationController?.abort();
      activePack = null;
      tokenizerPromise = null;
      await clearModelCache();
      post("cache-cleared");
    } else {
      throw new Error("Unknown worker operation: " + String(message.op));
    }
  } catch (error) {
    if (message.op !== "load" || error.message !== "Download cancelled.") {
      post("error", { message: error?.stack || error?.message || String(error) });
    }
  }
};
