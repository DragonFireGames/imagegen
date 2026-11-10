import { loadTokenizer, tokenizePrompt } from "./tokenizer.js";

function makeGaussianNoise(count, seed) {
  let state = (seed >>> 0) || 0x6d2b79f5;
  function random() {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  const output = new Float32Array(count);
  for (let i = 0; i < count; i += 2) {
    const u1 = Math.max(random(), 1e-12);
    const u2 = random();
    const radius = Math.sqrt(-2 * Math.log(u1));
    output[i] = radius * Math.cos(2 * Math.PI * u2);
    if (i + 1 < count) output[i + 1] = radius * Math.sin(2 * Math.PI * u2);
  }
  return output;
}

function makeIds(heightTokens, widthTokens) {
  const sequence = heightTokens * widthTokens;
  const imageIds = new BigInt64Array(sequence * 4);
  for (let y = 0; y < heightTokens; y++) {
    for (let x = 0; x < widthTokens; x++) {
      const index = (y * widthTokens + x) * 4;
      imageIds[index] = 0n;
      imageIds[index + 1] = BigInt(y);
      imageIds[index + 2] = BigInt(x);
      imageIds[index + 3] = 0n;
    }
  }

  const textIds = new BigInt64Array(512 * 4);
  for (let i = 0; i < 512; i++) textIds[i * 4 + 3] = BigInt(i);
  return { imageIds, textIds };
}

function runWorker(message, transfer, onProgress) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./inference.worker.js", import.meta.url), { type: "module" });
    let settled = false;

    const finish = (error, result) => {
      if (settled) return;
      settled = true;
      worker.terminate();
      if (error) reject(error);
      else resolve(result);
    };

    worker.onmessage = (event) => {
      const data = event.data;
      if (data.type === "status" || data.type === "step") onProgress(data);
      else if (data.type === "error") finish(new Error(data.message || "Inference worker failed."));
      else if (data.type === "done") finish(null, data);
      else finish(new Error("Unexpected worker response: " + String(data.type)));
    };
    worker.onerror = (event) => finish(new Error(event.message || "Inference worker crashed."));
    worker.onmessageerror = () => finish(new Error("Could not deserialize a message from the inference worker."));

    try {
      worker.postMessage(message, transfer || []);
    } catch (error) {
      finish(error);
    }
  });
}

export class Flux2Pipeline {
  constructor(onProgress = () => {}) {
    this.onProgress = onProgress;
    this.tokenizerPromise = null;
  }

  async generate({ prompt, width, height, steps, seed }) {
    const started = performance.now();
    const heightTokens = height / 16;
    const widthTokens = width / 16;
    if (!Number.isInteger(heightTokens) || !Number.isInteger(widthTokens)) {
      throw new Error("Width and height must be multiples of 16.");
    }

    this.onProgress({ type: "status", text: "Preparing tokenizer" });
    this.tokenizerPromise ||= loadTokenizer((text) => this.onProgress({ type: "status", text }));
    const tokenizer = await this.tokenizerPromise;
    const encoded = await tokenizePrompt(tokenizer, prompt);
    const inputIds = new BigInt64Array(512);
    const attentionMask = new BigInt64Array(512);
    for (let i = 0; i < 512; i++) {
      inputIds[i] = BigInt(Number(encoded.input_ids.data[i]));
      attentionMask[i] = BigInt(Number(encoded.attention_mask.data[i]));
    }

    this.onProgress({ type: "status", text: "Encoding prompt" });
    const textEncoded = await runWorker({
      op: "encode",
      inputIds: inputIds.buffer,
      attentionMask: attentionMask.buffer
    }, [inputIds.buffer, attentionMask.buffer], this.onProgress);

    const sequence = heightTokens * widthTokens;
    const latent = makeGaussianNoise(sequence * 128, seed);
    const { imageIds, textIds } = makeIds(heightTokens, widthTokens);

    const denoised = await runWorker({
      op: "denoise",
      height,
      width,
      steps,
      latent: latent.buffer,
      promptBuffer: textEncoded.buffer,
      promptDims: textEncoded.dims,
      imgIds: imageIds.buffer,
      txtIds: textIds.buffer
    }, [latent.buffer, textEncoded.buffer, imageIds.buffer, textIds.buffer], this.onProgress);

    const decoded = await runWorker({
      op: "decode",
      latent: denoised.latent,
      heightTokens,
      widthTokens
    }, [denoised.latent], this.onProgress);

    return {
      buffer: decoded.buffer,
      dims: decoded.dims,
      width,
      height,
      seed,
      steps,
      elapsed: (performance.now() - started) / 1000
    };
  }
}
