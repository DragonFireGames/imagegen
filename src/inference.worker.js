import * as ort from "../vendor/onnxruntime-web/ort.webgpu.min.mjs";

const MODEL_BASE = "https://huggingface.co/cgb/flux2-klein-4b-onnx-webgpu/resolve/main/";
ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";

function postStatus(text) {
  self.postMessage({ type: "status", text });
}

function bufferOf(view) {
  return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
}

async function createSession(modelFile, manifestFile, extraData = []) {
  const externalData = [...extraData];
  if (manifestFile) {
    postStatus("Reading " + manifestFile);
    const response = await fetch(MODEL_BASE + manifestFile);
    if (!response.ok) throw new Error("Could not fetch " + manifestFile + " (" + response.status + ")");
    const manifest = await response.json();
    for (const shard of manifest.shards || []) {
      externalData.push({ path: shard.path, data: MODEL_BASE + shard.path });
    }
  }

  postStatus("Downloading/loading " + modelFile + " from Hugging Face");
  return ort.InferenceSession.create(MODEL_BASE + modelFile, {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "all",
    externalData
  });
}

function halfToFloat(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 31;
  const fraction = value & 1023;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
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

function makeHalf(values) {
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = floatToHalf(values[i]);
  return out;
}

function asFloat32(tensor) {
  if (tensor.type === "float16" || tensor.data instanceof Uint16Array) {
    const out = new Float32Array(tensor.data.length);
    for (let i = 0; i < out.length; i++) out[i] = halfToFloat(tensor.data[i]);
    return out;
  }
  return Float32Array.from(tensor.data);
}

async function encode(message) {
  postStatus("Loading Qwen3 text encoder (about 2.2 GB)");
  let session;
  try {
    session = await createSession("text_encoder_q4.onnx", "text_encoder_q4.manifest.json");
    const inputIds = new BigInt64Array(message.inputIds);
    const attentionMask = new BigInt64Array(message.attentionMask);
    postStatus("Encoding prompt");
    const outputs = await session.run({
      input_ids: new ort.Tensor("int64", inputIds, [1, 512]),
      attention_mask: new ort.Tensor("int64", attentionMask, [1, 512])
    });
    const result = outputs.prompt_embeds || outputs[session.outputNames[0]];
    const data = bufferOf(result.data);
    self.postMessage({ type: "done", buffer: data, dims: result.dims, dtype: result.type }, [data]);
  } finally {
    if (session) await session.release();
  }
}

function schedulerSigmas(sequence, steps) {
  const mu = Math.max(0.5, Math.min(1.15, 0.5 + (1.15 - 0.5) * (sequence - 256) / (4096 - 256)));
  const expMu = Math.exp(mu);
  const sigmas = [];
  for (let i = 0; i <= steps; i++) {
    const sigma = 1 - i / steps;
    sigmas.push(sigma === 0 ? 0 : expMu / (expMu + (1 / sigma - 1)));
  }
  return sigmas;
}

async function denoise(message) {
  const heightTokens = message.height / 16;
  const widthTokens = message.width / 16;
  const sequence = heightTokens * widthTokens;
  const latent = new Float32Array(message.latent);
  const promptEmbeds = new Uint16Array(message.promptBuffer);
  const imgIds = new BigInt64Array(message.imgIds);
  const txtIds = new BigInt64Array(message.txtIds);
  const sigmas = schedulerSigmas(sequence, message.steps);
  let session;

  try {
    postStatus("Loading FLUX.2 transformer (about 4.1 GB)");
    session = await createSession("transformer_q8.onnx", "transformer_q8.manifest.json");
    const encoderTensor = new ort.Tensor("float16", promptEmbeds, message.promptDims);
    const imageIdTensor = new ort.Tensor("int64", imgIds, [1, sequence, 4]);
    const textIdTensor = new ort.Tensor("int64", txtIds, [1, 512, 4]);

    for (let step = 0; step < message.steps; step++) {
      self.postMessage({ type: "step", step: step + 1, total: message.steps });
      const hidden = new ort.Tensor("float16", makeHalf(latent), [1, sequence, 128]);
      const timestep = new ort.Tensor("float16", makeHalf([sigmas[step]]), [1]);
      const outputs = await session.run({
        hidden_states: hidden,
        encoder_hidden_states: encoderTensor,
        timestep,
        img_ids: imageIdTensor,
        txt_ids: textIdTensor
      });
      const prediction = outputs.noise_pred || outputs[session.outputNames[0]];
      const noise = asFloat32(prediction);
      if (noise.length !== latent.length) throw new Error("Transformer output shape did not match the latent input.");
      const delta = sigmas[step + 1] - sigmas[step];
      for (let i = 0; i < latent.length; i++) latent[i] += delta * noise[i];
    }

    const data = bufferOf(latent);
    self.postMessage({ type: "done", latent: data }, [data]);
  } finally {
    if (session) await session.release();
  }
}

async function decode(message) {
  const heightTokens = message.heightTokens;
  const widthTokens = message.widthTokens;
  const sequence = heightTokens * widthTokens;
  const tokens = new Float32Array(message.latent);

  if (tokens.length !== sequence * 128) {
    throw new Error(
      "VAE decoder expected " + (sequence * 128) +
      " latent values (" + sequence + " tokens × 128 features), got " + tokens.length + "."
    );
  }

  // The transformer returns [image_sequence, 128] tokens. The decoder
  // expects those same packed latents in channel-first NCHW order:
  // [1, 128, height/16, width/16]. Do not unpatchify them here; the
  // exported decoder graph consumes the 128 packed channels directly.
  const packed = new Float32Array(tokens.length);
  for (let token = 0; token < sequence; token++) {
    const tokenOffset = token * 128;
    for (let channel = 0; channel < 128; channel++) {
      packed[channel * sequence + token] = tokens[tokenOffset + channel];
    }
  }

  let session;
  try {
    postStatus("Loading VAE decoder");
    session = await createSession("vae_decoder.onnx", null, [
      { path: "vae_decoder.onnx.data", data: MODEL_BASE + "vae_decoder.onnx.data" }
    ]);
    const inputName = session.inputNames[0];
    const outputs = await session.run({
      [inputName]: new ort.Tensor("float32", packed, [1, 128, heightTokens, widthTokens])
    });
    const image = outputs.sample || outputs[session.outputNames[0]];
    const data = bufferOf(asFloat32(image));
    self.postMessage({ type: "done", buffer: data, dims: image.dims }, [data]);
  } finally {
    if (session) await session.release();
  }
}
self.onmessage = async (event) => {
  try {
    const message = event.data;
    if (message.op === "encode") await encode(message);
    else if (message.op === "denoise") await denoise(message);
    else if (message.op === "decode") await decode(message);
    else throw new Error("Unknown worker operation: " + message.op);
  } catch (error) {
    self.postMessage({ type: "error", message: error?.stack || error?.message || String(error) });
  }
};
