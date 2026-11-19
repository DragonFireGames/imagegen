// pixelsprite in the browser: the same sampler as pixelsprite/sampling.py, on onnxruntime-web.
//
// The ONNX graph (onnx/decoder.onnx in the Hugging Face repo) holds the 12 transformer blocks, the final norm and the
// head: one step with a key/value cache. Input vectors are sums of rows of small tables (onnx/embeddings.bin), built
// here exactly as in pixelsprite/model.py. Nothing in this file touches the page, so it also runs in Node (tests).

export const HF_REPO = "ihor-umanets/pixelsprite-canvas-v2";

export function modelFiles(base) {
  return {
    decoder: `${base}/onnx/decoder.onnx`,   // float32 graph, large weights as external data "decoder.data"
    decoderStatic: `${base}/onnx/decoder_static.onnx`,  // the same with a fixed-size cache (same external data)
    weights: `${base}/onnx/decoder.f16`,    // those weights as float16, in the same order
    embJson: `${base}/onnx/embeddings.json`,
    embBin: `${base}/onnx/embeddings.bin`,
    palette: `${base}/palette.json`,
  };
}

export const DATA_NAME = "decoder.data";

const json = (bytes) => JSON.parse(new TextDecoder().decode(bytes));

// On the CPU, onnxruntime-web computes inside session.run on this thread and resolves at once, so a loop of awaits
// never gives the browser a chance to paint (the page looks frozen; phones may give up on it). A MessageChannel
// task (no 4 ms clamp like setTimeout) lets it render between steps.
const yieldToBrowser = () => new Promise((resolve) => {
  const ch = new MessageChannel();
  ch.port1.onmessage = () => { ch.port1.close(); resolve(); };
  ch.port2.postMessage(0);
});

let HALF = null;
/** Little-endian float16 bytes -> float32 bytes (the weights are shipped as float16 to halve the download). */
export function halfToFloat(bytes) {
  if (!HALF) {
    HALF = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
      HALF[h] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
    }
  }
  if (bytes.byteOffset % 2) bytes = bytes.slice();
  const h = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1), out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) out[i] = HALF[h[i]];
  return new Uint8Array(out.buffer);
}

/** Loads the decoder. The weights go in as external data: onnxruntime-web then uploads them straight to the GPU
 * (WebGPU) instead of first copying the whole model into the WebAssembly heap, which never shrinks again; that
 * copy alone took ~560 MB and crashed phones. files.weights may be omitted for a self-contained graph (tests).
 * fixedCache: use decoder_static.onnx, whose cache keeps one size for the whole run (memory stays flat). */
export async function loadModel({ ort, files, ep = "wasm", fetchBytes, onProgress, sessionOptions = {}, fixedCache = false }) {
  const [info, pal, emb] = await Promise.all([
    fetchBytes(files.embJson).then(json), fetchBytes(files.palette).then(json), fetchBytes(files.embBin)]);
  const decoder = await fetchBytes(fixedCache ? files.decoderStatic : files.decoder, files.weights ? null : onProgress);
  const opts = { executionProviders: [ep], graphOptimizationLevel: "all", ...sessionOptions };
  if (files.weights) opts.externalData = [{ path: DATA_NAME, data: halfToFloat(await fetchBytes(files.weights, onProgress)) }];
  if (ep === "webgpu") {  // keep the cache on the GPU between steps; only the logits come back
    opts.preferredOutputLocation = {};
    for (let i = 0; i < info.n_layers; i++) {
      opts.preferredOutputLocation[`present_k_${i}`] = "gpu-buffer";
      opts.preferredOutputLocation[`present_v_${i}`] = "gpu-buffer";
    }
  }
  const session = await ort.InferenceSession.create(decoder, opts);
  const model = new SpriteModel(ort, session, info, emb, pal.colors, ep);
  model.fixedCache = fixedCache;
  return model;
}

// ------------------------------------------------------------------------------------------------- random numbers
export function makeRng(seed) {  // mulberry32: small, fast, seedable (the sprites differ from Python's for a seed)
  let a = (Number(seed) >>> 0) ^ 0x9e3779b9;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ------------------------------------------------------------------------------------------------------ guidance
function logSumExp(a, off, len) {
  let m = -Infinity;
  for (let i = 0; i < len; i++) if (a[off + i] > m) m = a[off + i];
  let s = 0;
  for (let i = 0; i < len; i++) s += Math.exp(a[off + i] - m);
  return m + Math.log(s);
}

function emptyLogOdds(l, off, K, E, colourLogSum = logSumExp(l, off, K)) {
  // log(pEmpty / (1 - pEmpty)) simplifies exactly to emptyLogit - logsumexp(colourLogits).
  // This avoids normalizing the full vocabulary and then converting the probability back to log-odds.
  return Math.min(l[off + E] - colourLogSum, Math.log(999999));
}

/** Guided distribution for one pixel. lc / lu: logits with and without text at offsets offC / offU (offU < 0 = no
 * guidance). Fills `col` (length K) with colour probabilities after temperature and top-p; returns p(EMPTY). */
export function guided(lc, offC, lu, offU, K, gShape, gColour, temperature, topP, col, order) {
  const E = K;
  let s;
  if (offU >= 0) {
    const lsC = logSumExp(lc, offC, K), lsU = logSumExp(lu, offU, K);
    const sc = emptyLogOdds(lc, offC, K, E, lsC), su = emptyLogOdds(lu, offU, K, E, lsU);
    s = su + gShape * (sc - su);
    for (let j = 0; j < K; j++) {
      const u = lu[offU + j] - lsU;
      col[j] = u + gColour * ((lc[offC + j] - lsC) - u);
    }
  } else {
    const lsC = logSumExp(lc, offC, K);
    s = emptyLogOdds(lc, offC, K, E, lsC);
    for (let j = 0; j < K; j++) col[j] = lc[offC + j];
  }
  finishColour(col, K, temperature, topP, order);
  return 1 / (1 + Math.exp(-s));
}

/** Mixing k captions (AND / product of experts): rows base .. base+k-1 hold the captions, row base+k the "no text"
 * condition. w[g] is caption g's weight at this pixel. Shape log-odds and colour log-probabilities are
 * s = s_u + sum_g w_g (s_g - s_u); then temperature and top-p as in guided(). */
export function guidedMix(L, base, k, K, w, temperature, topP, col, order) {
  const E = K, V = K + 1, offU = (base + k) * V;
  const lsU = logSumExp(L, offU, K), su = emptyLogOdds(L, offU, K, E, lsU);
  let s = su;
  for (let j = 0; j < K; j++) col[j] = L[offU + j] - lsU;
  for (let g = 0; g < k; g++) {
    const off = (base + g) * V, wg = w[g];
    if (!wg) continue;
    const ls = logSumExp(L, off, K);
    s += wg * (emptyLogOdds(L, off, K, E, ls) - su);
    for (let j = 0; j < K; j++) col[j] += wg * ((L[off + j] - ls) - (L[offU + j] - lsU));
  }
  finishColour(col, K, temperature, topP, order);
  return 1 / (1 + Math.exp(-s));
}

/** Colour log-scores -> probabilities: temperature, softmax, then top-p. */
function finishColour(col, K, temperature, topP, order) {
  const inv = 1 / Math.max(temperature, 1e-6);
  let m = -Infinity;
  for (let j = 0; j < K; j++) { col[j] *= inv; if (col[j] > m) m = col[j]; }
  let sum = 0;
  for (let j = 0; j < K; j++) { col[j] = Math.exp(col[j] - m); sum += col[j]; }
  for (let j = 0; j < K; j++) col[j] /= sum;
  if (topP < 1) {  // nucleus: keep the most likely colours whose mass before them is < topP (always the first)
    for (let j = 0; j < K; j++) order[j] = j;
    order.sort((a, b) => col[b] - col[a]);
    let cum = 0, kept = 0;
    for (let r = 0; r < K; r++) {
      const j = order[r];
      if (cum < topP) { cum += col[j]; kept += col[j]; } else col[j] = 0;
    }
    for (let j = 0; j < K; j++) col[j] /= kept;
  }
}

function draw(col, K, u) {
  let acc = 0;
  for (let j = 0; j < K; j++) { acc += col[j]; if (u < acc) return j; }
  for (let j = K - 1; j >= 0; j--) if (col[j] > 0) return j;  // rounding at the very end
  return 0;
}

// ------------------------------------------------------------------------------------------ inference strategies
// The runner (SpriteModel.drawChunk) only batches network rows, runs the model one token at a time and feeds each
// sprite's pixel back into its rows. HOW a sprite turns the logits of its rows into its next pixel is a strategy:
//
//   { name,
//     rows: [condition vectors from model.condition(...)],  // one network row each, contiguous in the batch
//     pixel(L, base, t, c) -> p(EMPTY) }  // L: logits of the chunk (row r starts at r * (K + 1)); base: this
//                                         // sprite's first row; t: pixel index; c: {K, temperature, topP, col,
//                                         // order}; fills c.col with the colour probabilities
//
// Sprites with different strategies can share a chunk. To add a new way of sampling, add a factory here (or pass
// your own object to generate({strategyFor})); the runner does not change. Cost: one network row per entry of rows.
export const Strategy = {
  /** One caption (or none), no guidance: the model's own distribution. */
  plain(model, vector) {
    return {
      name: "plain", rows: [model.condition(vector)],
      pixel: (L, base, t, c) => guided(L, base * (c.K + 1), L, -1, c.K, 1, 1, c.temperature, c.topP, c.col, c.order),
    };
  },
  /** Split classifier-free guidance: the shape and the colours pulled towards the caption with their own strengths
   * (needs a "no text" row as well). Without a caption, or with both strengths 1, it is plain. */
  split(model, vector, gShape = 1, gColour = 2) {
    if (!vector || (gShape === 1 && gColour === 1)) return Strategy.plain(model, vector);
    const V = model.K + 1;
    return {
      name: "split", rows: [model.condition(vector), model.condition(null)],
      pixel: (L, base, t, c) => guided(L, base * V, L, (base + 1) * V, c.K, gShape, gColour, c.temperature, c.topP, c.col, c.order),
    };
  },
  /** Several captions at once (AND / product of experts) with weights per pixel: maps[g][t] for caption g. */
  mix(model, vectors, maps) {
    const k = vectors.length, w = new Float64Array(k);
    return {
      name: "mix", rows: [...vectors.map((v) => model.condition(v)), model.condition(null)],
      pixel: (L, base, t, c) => {
        for (let g = 0; g < k; g++) w[g] = maps[g][t];
        return guidedMix(L, base, k, c.K, w, c.temperature, c.topP, c.col, c.order);
      },
    };
  },
};

// ------------------------------------------------------------------------------------------------ silhouettes
/** Edge flag per pixel: 0 inside, 1 on the boundary (4-neighbourhood, canvas edge = outside), 2 outside,
 * 3 unknown (no silhouette given). */
export function edgeFlags(mask, S) {
  const e = new Int8Array(S * S);
  if (!mask) return e.fill(3);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const p = y * S + x;
    if (!mask[p]) { e[p] = 2; continue; }
    const inner = y > 0 && mask[p - S] && y < S - 1 && mask[p + S] && x > 0 && mask[p - 1] && x < S - 1 && mask[p + 1];
    e[p] = inner ? 0 : 1;
  }
  return e;
}

/** Sizes of the 8-connected pieces of the non-transparent pixels, largest first. */
export function pieces(img, S) {
  const seen = new Uint8Array(S * S), sizes = [], q = new Int32Array(S * S);
  for (let p0 = 0; p0 < S * S; p0++) {
    if (img[p0] < 0 || seen[p0]) continue;
    let head = 0, tail = 0, n = 0;
    q[tail++] = p0; seen[p0] = 1;
    while (head < tail) {
      const p = q[head++]; n++;
      const y = (p / S) | 0, x = p % S;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx;
        if (yy < 0 || xx < 0 || yy >= S || xx >= S) continue;
        const pp = yy * S + xx;
        if (img[pp] >= 0 && !seen[pp]) { seen[pp] = 1; q[tail++] = pp; }
      }
    }
    sizes.push(n);
  }
  return sizes.sort((a, b) => b - a);
}

/** Reason to draw a sprite again (empty canvas or a detached piece of 1-2 px), or null. */
export function rejected(img, S, maxPx = 2) {
  const sizes = pieces(img, S);
  if (!sizes.length) return "empty";
  if (sizes.length >= 2 && Math.min(...sizes.slice(1)) <= maxPx) return "detached_1_2px";
  return null;
}

// ---------------------------------------------------------------------------------------------------- the model
export class SpriteModel {
  constructor(ort, session, info, embBytes, palette, ep) {
    this.ort = ort; this.session = session; this.info = info; this.ep = ep;
    const buf = embBytes.slice().buffer;  // own, aligned copy
    this.t = {};
    for (const [name, v] of Object.entries(info.tables))
      this.t[name] = new Float32Array(buf, v.offset, v.shape.reduce((a, b) => a * b, 1));
    this.d = info.d_model; this.K = info.n_colors; this.E = info.empty; this.BOS = info.bos;
    this.L = info.n_layers; this.H = info.n_heads; this.hd = this.d / this.H;
    this.textDim = info.tables["cond_proj.weight"].shape[1];
    this.palette = Uint8Array.from(palette.flat());
  }

  _add(dst, o, name, i) { const d = this.d, t = this.t[name]; for (let j = 0; j < d; j++) dst[o + j] += t[i * d + j]; }

  /** Caption vector (Float32Array, L2-normalised) -> condition vector; null -> the learned "no text" vector. */
  condition(text) {
    if (!text) return this.t.null_cond.slice();
    const W = this.t["cond_proj.weight"], b = this.t["cond_proj.bias"], D = this.textDim, out = new Float32Array(this.d);
    for (let o = 0; o < this.d; o++) {
      let s = b[o];
      for (let i = 0; i < D; i++) s += W[o * D + i] * text[i];
      out[o] = s;
    }
    return out;
  }

  /** Writes the 2 + S prefix vectors (text, size, one per silhouette row) at dst[o...]. */
  writePrefix(dst, o, cond, S, mask) {
    const d = this.d, k = this.info.kinds, M = this.info.max_size;
    dst.set(cond, o); this._add(dst, o, "kind", k.text);
    o += d;
    dst.fill(0, o, o + d); this._add(dst, o, "size_emb", this.info.sizes.indexOf(S)); this._add(dst, o, "kind", k.size);
    const W = this.t["mask_proj.weight"], bias = this.t["mask_proj.bias"];
    for (let r = 0; r < S; r++) {
      o += d;
      dst.set(bias, o);
      this._add(dst, o, "row", r);
      this._add(dst, o, "kind", mask ? k.row_known : k.row_unknown);
      if (mask) for (let c = 0; c < S; c++) if (mask[r * S + c]) for (let j = 0; j < d; j++) dst[o + j] += W[j * M + c];
    }
  }

  /** Writes one pixel token: previous pixel's token + row + column + edge flag of this pixel. */
  writePixel(dst, o, prev, r, c, e) {
    // This runs for every batch row at every generated pixel. Walk the embedding
    // dimension once instead of making five separate passes through the vector.
    // Keep the original addition order and Float32 rounding exactly.
    const d = this.d, tok = this.t.tok, row = this.t.row, col = this.t.col;
    const edge = this.t.edge, kind = this.t.kind;
    const to = prev * d, ro = r * d, co = c * d, eo = e * d, ko = this.info.kinds.pixel * d;
    for (let j = 0; j < d; j++) {
      const p = o + j;
      dst[p] = tok[to + j];
      dst[p] += row[ro + j];
      dst[p] += col[co + j];
      dst[p] += edge[eo + j];
      dst[p] += kind[ko + j];
    }
  }

  emptyCache(B) {
    const z = new Float32Array(0), out = [];
    for (let i = 0; i < 2 * this.L; i++) out.push(new this.ort.Tensor("float32", z, [B, this.H, 0, this.hd]));
    return out;
  }

  /** One decoder call: x (B*T*d) + cache -> logits of the last token (B*(K+1)) and the longer cache. */
  async step(x, B, T, cache) {
    const feeds = { x: new this.ort.Tensor("float32", x, [B, T, this.d]) };
    for (let i = 0; i < this.L; i++) { feeds[`past_k_${i}`] = cache[2 * i]; feeds[`past_v_${i}`] = cache[2 * i + 1]; }
    const out = await this.session.run(feeds);
    const next = [];
    for (let i = 0; i < this.L; i++) next.push(out[`present_k_${i}`], out[`present_v_${i}`]);
    return { logits: out.logits.data, cache: next };
  }

  static release(cache) { for (const t of cache) if (t.location === "gpu-buffer") t.dispose(); }

  /** A zero-filled fixed-size cache: 2 * layers tensors (B, H, M, hd). On WebGPU they are created on the GPU
   * (new buffers start as zeros): a CPU tensor would first be copied into the WebAssembly heap, which never shrinks
   * (256 MB after one 16x16 run of 8 sprites, 531 MB after 32x32). */
  async cacheSet(B, M) {
    const dims = [B, this.H, M, this.hd], out = [];
    if (this.ep === "webgpu") {
      const device = await this.ort.env.webgpu.device, size = B * this.H * M * this.hd * 4;
      for (let i = 0; i < 2 * this.L; i++) {
        const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        out.push(this.ort.Tensor.fromGpuBuffer(buf, { dataType: "float32", dims, dispose: () => buf.destroy() }));
      }
      return out;
    }
    for (let i = 0; i < 2 * this.L; i++) out.push(new this.ort.Tensor("float32", new Float32Array(B * this.H * M * this.hd), dims));
    return out;
  }

  /** One call of decoder_static.onnx: token x (B*d) at position pos. (No preallocated outputs: for CPU tensors
   * onnxruntime-web 1.31 hands back the same objects but leaves their data untouched, so the cache stayed zero and
   * the model saw only the current pixel: horizontal stripes on phones.) */
  async stepFixed(x, B, pos, cache, reusable = null) {
    const feeds = reusable ? reusable.feeds : {
      x: new this.ort.Tensor("float32", x, [B, 1, this.d]),
      pos: new this.ort.Tensor("int64", BigInt64Array.of(BigInt(pos)), [1]),
    };
    if (reusable) reusable.position[0] = BigInt(pos);
    for (let i = 0; i < this.L; i++) { feeds[`past_k_${i}`] = cache[2 * i]; feeds[`past_v_${i}`] = cache[2 * i + 1]; }
    const out = await this.session.run(feeds);
    const next = [];
    for (let i = 0; i < this.L; i++) next.push(out[`present_k_${i}`], out[`present_v_${i}`]);
    return { logits: out.logits.data, cache: next };
  }

  /** The runner: draws one sprite per strategy on an S x S canvas in parallel. Every strategy owns rows.length
   * contiguous rows of the batch; the runner knows nothing about guidance. Returns Int16Arrays (palette index,
   * -1 = transparent), one per strategy. */
  async drawChunk(strategies, p, rng, onPixel) {
    const { S, mask, win, temperature, topP } = p;
    const b = strategies.length, n = S * S, d = this.d, K = this.K, E = this.E;
    const bases = [];
    let B = 0;
    for (const st of strategies) { bases.push(B); B += st.rows.length; }
    const owner = new Int32Array(B);  // row -> sprite
    strategies.forEach((st, i) => { for (let r = 0; r < st.rows.length; r++) owner[bases[i] + r] = i; });
    const edge = edgeFlags(mask, S);
    const T0 = 2 + S + 1, x0 = new Float32Array(B * T0 * d);
    for (let j = 0; j < B; j++) {
      const i = owner[j];
      this.writePrefix(x0, j * T0 * d, strategies[i].rows[j - bases[i]], S, mask);
      this.writePixel(x0, (j * T0 + T0 - 1) * d, this.BOS, 0, 0, edge[0]);
    }
    let logits, cache, pos = 0, lastYield = performance.now();
    const x = new Float32Array(B * d);
    let fixedStepState = null;
    if (this.fixedCache) {
      const position = new BigInt64Array(1);
      fixedStepState = {
        position,
        feeds: {
          x: new this.ort.Tensor("float32", x, [B, 1, d]),
          pos: new this.ort.Tensor("int64", position, [1]),
        },
      };
    }
    const advance = async () => {  // one more token -> logits of the next pixel
      const prev = cache;
      const r = this.fixedCache ? await this.stepFixed(x, B, pos, cache, fixedStepState) : await this.step(x, B, 1, cache);
      pos++;
      logits = r.logits; cache = r.cache;
      if (prev !== cache) SpriteModel.release(prev);
    };
    if (this.fixedCache) {  // fixed-size cache: reuse input tensors and feed objects for every token
      cache = await this.cacheSet(B, 2 + S + n);
      for (let i = 0; i < T0; i++) {
        for (let j = 0; j < B; j++) x.set(x0.subarray((j * T0 + i) * d, (j * T0 + i + 1) * d), j * d);
        await advance();
      }
    } else ({ logits, cache } = await this.step(x0, B, T0, this.emptyCache(B)));
    const out = Array.from({ length: b }, () => new Int16Array(n).fill(-1));
    const tok = new Int32Array(b);
    const ctx = { K, temperature, topP, col: new Float64Array(K), order: topP < 1 ? new Int32Array(K) : null };
    try {
      for (let t = 0; t < n; t++) {
        if (p.signal && p.signal.aborted) { const e = new Error("stopped"); e.name = "AbortError"; throw e; }
        const forceEmpty = !win[t] || (mask && !mask[t]), forceFill = mask && mask[t];
        for (let i = 0; i < b; i++) {
          let pe = 0;
          if (forceEmpty) pe = 1;
          else {
            pe = strategies[i].pixel(logits, bases[i], t, ctx);
            if (forceFill) pe = 0;
          }
          tok[i] = rng() < pe ? E : draw(ctx.col, K, rng());
          out[i][t] = tok[i] === E ? -1 : tok[i];
        }
        if (onPixel) onPixel(t, out);
        if (t + 1 === n) break;
        const r = ((t + 1) / S) | 0, c = (t + 1) % S;
        for (let j = 0; j < B; j++) this.writePixel(x, j * d, tok[owner[j]], r, c, edge[t + 1]);
        await advance();
        if (performance.now() - lastYield > 40) { await yieldToBrowser(); lastYield = performance.now(); }
      }
    } finally {
      SpriteModel.release(cache);
    }
    return out;
  }

  /** n sprites of size 16, 24 or 32 (24 = a 24x24 window in the middle of the 32x32 canvas).
   * vector: caption vector or null; mask: Uint8Array(size*size) silhouette or null (then the filter is off).
   * onPixel(ids, t, images, view) is called after every pixel; view = {S, o, w}: canvas size, window offset and
   * width. signal: an AbortSignal to stop between pixels. Returns {images: Int16Array(size*size)[], stats}.
   * How each sprite is sampled: strategyFor(id, view) -> a Strategy (see above); without it, split guidance on
   * `vector` with gShape / gColour for every sprite. Chunks hold as many sprites as fit in `rows` network rows
   * (default: `chunk` sprites of the first sprite's strategy, i.e. 2 * chunk rows for split guidance). */
  async generate({ vector = null, n = 8, size = 16, seed = 0, gShape = 1, gColour = 2, temperature = 1, topP = 1,
                   mask = null, filter = true, chunk = 8, rows = 0, maxRounds = 50, onPixel = null, onSprite = null,
                   signal = null, strategyFor = null }) {
    if (![16, 24, 32].includes(size)) throw new Error("size must be 16, 24 or 32");
    const S = size === 24 ? 32 : size, w = size, o = (S - w) >> 1;
    let m = null;
    if (mask) {
      m = new Uint8Array(S * S);
      for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) m[(y + o) * S + x + o] = mask[y * w + x] ? 1 : 0;
    }
    const win = new Uint8Array(S * S);
    for (let y = o; y < o + w; y++) for (let x = o; x < o + w; x++) win[y * S + x] = 1;
    const useFilter = filter && !mask;
    const view = { S, o, w };
    const shared = strategyFor ? null : Strategy.split(this, vector, gShape, gColour);
    const strategyOf = (id) => (strategyFor ? strategyFor(id, view) : shared);
    const p = { S, mask: m, win, temperature, topP, signal };
    const rng = makeRng(seed);
    const images = new Array(n), stats = { sampled: 0, rejected: 0, rounds: 0, reasons: {} };
    let pending = Array.from({ length: n }, (_, i) => i);
    while (pending.length && stats.rounds < maxRounds) {
      stats.rounds++;
      const again = [];
      for (let s = 0; s < pending.length;) {
        const ids = [], sts = [];  // pack sprites while their rows fit the budget (at least one sprite per chunk)
        let budget = 0, used = 0;
        while (s < pending.length) {
          const st = strategyOf(pending[s]);
          if (!budget) budget = rows || chunk * st.rows.length;
          if (ids.length && used + st.rows.length > budget) break;
          ids.push(pending[s]); sts.push(st); used += st.rows.length; s++;
        }
        const got = await this.drawChunk(sts, p, rng, onPixel ? (t, imgs) => onPixel(ids, t, imgs, view) : null);
        ids.forEach((id, k) => {
          stats.sampled++;
          const why = useFilter ? rejected(got[k], S) : null;
          if (why) { stats.rejected++; stats.reasons[why] = (stats.reasons[why] || 0) + 1; again.push(id); return; }
          images[id] = crop(got[k], S, o, w);
          if (onSprite) onSprite(id, images[id]);
        });
      }
      pending = again;
    }
    if (pending.length) throw new Error(`${pending.length} sprites still rejected after ${maxRounds} rounds`);
    return { images, size, stats };
  }

  /** Palette-index image -> RGBA bytes (alpha 0 or 255). */
  rgba(img) {
    const out = new Uint8ClampedArray(img.length * 4);
    for (let p = 0; p < img.length; p++) {
      const k = img[p];
      if (k < 0) continue;
      out[4 * p] = this.palette[3 * k]; out[4 * p + 1] = this.palette[3 * k + 1]; out[4 * p + 2] = this.palette[3 * k + 2];
      out[4 * p + 3] = 255;
    }
    return out;
  }
}

function crop(img, S, o, w) {
  if (w === S) return img;
  const out = new Int16Array(w * w);
  for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) out[y * w + x] = img[(y + o) * S + x + o];
  return out;
}
