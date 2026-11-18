// Caption -> 384-number vector in the browser: the same multilingual sentence encoder the model was trained with
// (paraphrase-multilingual-MiniLM-L12-v2), as the int8 ONNX export by Xenova, mean pooling + L2 normalisation
// like sentence-transformers. Loaded only when someone types their own caption (about 120 MB, cached afterwards).

export const ENCODER = {
  repo: "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
  revision: "2c4055b12046f11709e9df2c122e59ffbdc2f900",
  model: "onnx/model_quantized.onnx",
  maxTokens: 128,
};

export function encoderFiles(base) {
  return { model: `${base}/${ENCODER.model}`, tokenizer: `${base}/tokenizer.json`, config: `${base}/tokenizer_config.json` };
}

const json = (bytes) => JSON.parse(new TextDecoder().decode(bytes));

export async function loadTextEncoder({ ort, Tokenizer, files, fetchBytes, onProgress }) {
  const [tj, tc] = await Promise.all([fetchBytes(files.tokenizer).then(json), fetchBytes(files.config).then(json)]);
  const tokenizer = new Tokenizer(tj, tc);
  const bytes = await fetchBytes(files.model, onProgress);
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  const cache = new Map();

  async function encode(text) {
    if (cache.has(text)) return cache.get(text);
    let ids = tokenizer.encode(text).ids;
    if (ids.length > ENCODER.maxTokens) ids = [...ids.slice(0, ENCODER.maxTokens - 1), ids[ids.length - 1]];
    const n = ids.length, shape = [1, n];
    const feeds = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), shape),
      attention_mask: new ort.Tensor("int64", new BigInt64Array(n).fill(1n), shape),
    };
    if (session.inputNames.includes("token_type_ids")) feeds.token_type_ids = new ort.Tensor("int64", new BigInt64Array(n), shape);
    const out = await session.run(feeds);
    const h = out[session.outputNames[0]].data, D = h.length / n, v = new Float32Array(D);
    for (let t = 0; t < n; t++) for (let j = 0; j < D; j++) v[j] += h[t * D + j] / n;
    let norm = 0;
    for (let j = 0; j < D; j++) norm += v[j] * v[j];
    norm = Math.sqrt(norm) || 1;
    for (let j = 0; j < D; j++) v[j] /= norm;
    cache.set(text, v);
    return v;
  }
  return { encode, tokenizer };
}
