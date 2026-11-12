const MODEL_BASE = "https://huggingface.co/aaaaaeeeee/blip-sdxl-anime-wai-illustrious17/resolve/main/";
const SCHEDULER_URL = "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/b41f1c324e32e03f9e104bd9ace0992e4e135597/scheduler/scheduler_config.json";
const SCHEDULER_CACHE_PATH = "@scheduler/scheduler_config.json";
const CACHE_NAME = "illustrious-wai-v17-model-v1";
const BUNDLE_MAGIC = "WKIMDL01";
const CACHE_PREFIX = "/__illustrious_model_cache__/";

function resourcePath(url) {
  if (typeof url !== "string") return null;
  if (url === SCHEDULER_URL) return SCHEDULER_CACHE_PATH;
  if (!url.startsWith(MODEL_BASE)) return null;
  return decodeURIComponent(url.slice(MODEL_BASE.length).split("?")[0]);
}

function cacheKey(path) {
  return new Request(new URL(CACHE_PREFIX + encodeURIComponent(path), location.origin).href);
}

export function isModelUrl(url) {
  return resourcePath(url) !== null;
}

function reportCacheMiss(path, callback, reason = "not cached") {
  const message = "MODEL CACHE MISS · " + path + " · " + reason + "; downloading a fallback copy.";
  if (typeof callback === "function") callback(path, reason);
  else console.warn("[Illustrious model cache] " + message);
}

export async function fetchModelResource(url, options = {}) {
  const { onCacheMiss, expectedSize, forceRefresh, ...requestOptions } = options;
  const path = resourcePath(url);
  if (!path) return fetch(url, requestOptions);

  let cache = null;
  let cacheFailure = "persistent cache entry is missing";
  const key = cacheKey(path);
  if ("caches" in globalThis) {
    try {
      cache = await caches.open(CACHE_NAME);
      if (!forceRefresh) {
        const saved = await cache.match(key);
        if (saved) {
          if (Number.isSafeInteger(expectedSize)) {
            const actualSize = (await saved.clone().blob()).size;
            if (actualSize === expectedSize) return saved;
            await cache.delete(key);
            cacheFailure = "cached size mismatch (" + actualSize + " vs " + expectedSize + ")";
          } else {
            return saved;
          }
        }
      } else {
        cacheFailure = "explicit refresh requested";
      }
    } catch (error) {
      cacheFailure = "Cache Storage read failed: " + (error?.message || error);
      cache = null;
    }
  } else {
    cacheFailure = "Cache Storage is unavailable";
  }

  reportCacheMiss(path, onCacheMiss, cacheFailure);
  const fetchOptions = { ...requestOptions, credentials: "omit" };
  if (forceRefresh) fetchOptions.cache = "no-cache";
  const response = await fetch(url, fetchOptions);
  if (!response.ok) return response;

  if (Number.isSafeInteger(expectedSize)) {
    const actualSize = (await response.clone().blob()).size;
    if (actualSize !== expectedSize) {
      throw new Error(path + " size mismatch during fallback download: expected " +
        expectedSize + ", got " + actualSize + ".");
    }
  }

  if (cache) {
    try {
      await cache.put(key, response.clone());
    } catch (error) {
      console.warn("[Illustrious model cache] Could not cache fallback " + path + ": " + (error?.message || error));
    }
  }
  return response;
}

function generationPaths(manifest, includeVaeEncoder = false) {
  const paths = new Set(["manifest.json"]);
  const componentNames = ["text_encoder", "text_encoder_2", "unet_a", "unet_b", "unet_c", "vae_decoder"];
  if (includeVaeEncoder && manifest.components?.vae_encoder) componentNames.push("vae_encoder");
  for (const name of componentNames) {
    const component = manifest.components?.[name];
    if (!component) throw new Error("Model manifest is missing " + name + ".");
    if (component.onnx) paths.add(component.onnx);
    if (component.weightsIndex) paths.add(component.weightsIndex);
    for (const shard of component.shards || []) paths.add(shard);
  }
  for (const folder of ["tokenizer", "tokenizer_2"]) {
    for (const file of ["vocab.json", "merges.txt", "special_tokens_map.json"]) paths.add(folder + "/" + file);
  }
  return [...paths];
}

function fileRecord(manifest, path) {
  return (manifest.files || []).find(item => item.path === path) || null;
}

async function verifyBlob(blob, record, path) {
  if (record?.size !== undefined && blob.size !== record.size) {
    throw new Error(path + " size mismatch: expected " + record.size + ", got " + blob.size + ".");
  }
  if (record?.sha256 && blob.size <= 256 * 1024 * 1024) {
    const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
    const actual = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, "0")).join("");
    if (actual !== record.sha256) throw new Error(path + " SHA-256 mismatch.");
  }
}

async function sha256Blob(blob) {
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, "0")).join("");
}

async function validSchedulerResponse(response) {
  try {
    const config = await response.clone().json();
    return Number.isInteger(config.num_train_timesteps) &&
      config.beta_schedule === "scaled_linear" &&
      config.prediction_type === "epsilon" &&
      config.timestep_spacing === "leading";
  } catch {
    return false;
  }
}

export async function prepareModelCache(manifest, log, progress, checkCancelled, includeVaeEncoder = false) {
  if (!("caches" in globalThis)) throw new Error("This browser does not support persistent Cache Storage.");
  const paths = generationPaths(manifest, includeVaeEncoder);
  const records = new Map((manifest.files || []).map(item => [item.path, item]));
  const requiredBytes = paths.reduce((sum, path) => sum + Number(records.get(path)?.size || 0), 0);
  const estimate = await navigator.storage?.estimate?.();
  if (estimate?.quota && estimate.quota - estimate.usage < requiredBytes * 1.05) {
    throw new Error("Insufficient browser storage for the model cache: need about " +
      (requiredBytes / 1024 ** 3).toFixed(2) + " GiB; available " +
      ((estimate.quota - estimate.usage) / 1024 ** 3).toFixed(2) + " GiB. Free browser storage or use the model bundle import/export controls.");
  }
  try { await navigator.storage?.persist?.(); } catch {}
  const cache = await caches.open(CACHE_NAME);
  let completed = 0, reused = 0, downloaded = 0;
  const total = paths.length;
  let cursor = 0;
  const workers = Array.from({ length: 2 }, async () => {
    while (true) {
      const index = cursor++;
      if (index >= paths.length) return;
      checkCancelled();
      const path = paths[index], key = cacheKey(path), record = records.get(path);
      let response = await cache.match(key);
      if (response) {
        const blob = await response.blob();
        try { await verifyBlob(blob, record, path); reused++; }
        catch { await cache.delete(key); response = null; }
      }
      if (!response) {
        const network = await fetch(MODEL_BASE + path, { credentials: "omit", cache: "force-cache" });
        if (!network.ok) throw new Error("Model download failed for " + path + " · HTTP " + network.status);
        const blob = await network.blob();
        await verifyBlob(blob, record, path);
        response = new Response(blob, { headers: { "Content-Type": "application/octet-stream" } });
        await cache.put(key, response.clone());
        downloaded++;
      }
      completed++;
      progress?.(completed / total * 100, path);
      if (completed === 1 || completed === total || completed % 5 === 0) {
        log("Model cache " + completed + "/" + total + " · " +
          (downloaded ? downloaded + " downloaded" : "checking") +
          " · " + reused + " reused · " + path);
      }
    }
  });
  await Promise.all(workers);

  const schedulerKey = cacheKey(SCHEDULER_CACHE_PATH);
  let schedulerResponse = await cache.match(schedulerKey);
  if (schedulerResponse && !(await validSchedulerResponse(schedulerResponse))) {
    await cache.delete(schedulerKey);
    schedulerResponse = null;
  }
  if (schedulerResponse) {
    reused++;
  } else {
    const response = await fetch(SCHEDULER_URL, { credentials: "omit", cache: "force-cache" });
    if (!response.ok) throw new Error("Scheduler config download failed · HTTP " + response.status);
    if (!(await validSchedulerResponse(response))) throw new Error("Scheduler config download was invalid or incompatible.");
    await cache.put(schedulerKey, response.clone());
    downloaded++;
  }

  const totalFiles = paths.length + 1;
  log("Model cache ready · " + totalFiles + " files including scheduler · " + downloaded +
    " downloaded · " + reused + " reused · " + (requiredBytes / 1024 ** 3).toFixed(2) + " GiB verified.");
  return { paths, bytes: requiredBytes, downloaded, reused, totalFiles };
}

export async function exportModelBundle(manifest, log) {
  const paths = [...generationPaths(manifest, true), SCHEDULER_CACHE_PATH];
  const cache = await caches.open(CACHE_NAME);
  const entries = [];
  const blobs = [];
  let offset = 0;
  for (const path of paths) {
    const response = await cache.match(cacheKey(path));
    if (!response) throw new Error("Model cache is incomplete; missing " + path + ". Prepare the cache first.");
    const blob = await response.blob();
    const record = path === SCHEDULER_CACHE_PATH ? null : fileRecord(manifest, path);
    await verifyBlob(blob, record, path);
    entries.push({
      path,
      offset,
      size: blob.size,
      sha256: record?.sha256 || (path === SCHEDULER_CACHE_PATH ? await sha256Blob(blob) : null)
    });
    blobs.push(blob);
    offset += blob.size;
  }
  const header = new TextEncoder().encode(JSON.stringify({ version: 1, model: "sdxl-anime-wai-illustrious17", entries }));
  const prefix = new ArrayBuffer(12);
  new Uint8Array(prefix).set(new TextEncoder().encode(BUNDLE_MAGIC));
  new DataView(prefix).setUint32(8, header.length, true);
  const bundle = new Blob([prefix, header, ...blobs], { type: "application/octet-stream" });
  log("Created model bundle · " + entries.length + " files including VAE encoder and scheduler · " +
    (bundle.size / 1024 ** 3).toFixed(2) + " GiB.");
  return bundle;
}

export async function importModelBundle(file, manifest, log, progress) {
  if (!file || file.size < 12) throw new Error("Choose a valid .wkimodel bundle.");
  const prefix = new Uint8Array(await file.slice(0, 12).arrayBuffer());
  if (new TextDecoder().decode(prefix.slice(0, 8)) !== BUNDLE_MAGIC) throw new Error("Not a WatermelonKatana Illustrious model bundle.");
  const headerLength = new DataView(prefix.buffer).getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(await file.slice(12, 12 + headerLength).arrayBuffer()));
  if (header.version !== 1 || header.model !== "sdxl-anime-wai-illustrious17" || !Array.isArray(header.entries)) {
    throw new Error("Unsupported model bundle version or model identifier.");
  }
  const required = new Set(generationPaths(manifest));
  const allKnown = new Set(generationPaths(manifest, true));
  const supplied = new Set(header.entries.map(entry => entry.path));
  for (const path of required) if (!supplied.has(path)) throw new Error("Model bundle is incomplete; missing " + path + ".");
  const cache = await caches.open(CACHE_NAME);
  const records = new Map((manifest.files || []).map(item => [item.path, item]));
  let done = 0;
  for (const entry of header.entries) {
    const isScheduler = entry.path === SCHEDULER_CACHE_PATH;
    const record = isScheduler ? null : records.get(entry.path);
    if (!allKnown.has(entry.path) && !isScheduler) throw new Error("Model bundle contains an unexpected file " + entry.path + ".");
    if (!isScheduler && (!record || entry.size !== record.size ||
        (record.sha256 && entry.sha256 !== record.sha256))) {
      throw new Error("Bundle manifest does not match current model file " + entry.path + ".");
    }
    const start = 12 + headerLength + entry.offset;
    const blob = file.slice(start, start + entry.size);
    if (isScheduler) {
      if (!entry.sha256 || await sha256Blob(blob) !== entry.sha256 ||
          !(await validSchedulerResponse(new Response(blob)))) {
        throw new Error("Bundle scheduler config failed integrity or compatibility checks.");
      }
    } else {
      await verifyBlob(blob, record, entry.path);
    }
    await cache.put(cacheKey(entry.path), new Response(blob, { headers: { "Content-Type": "application/octet-stream" } }));
    done++;
    progress?.(done / header.entries.length * 100, entry.path);
    if (done === 1 || done === header.entries.length || done % 5 === 0) log("Imported " + done + "/" + header.entries.length + " model files · " + entry.path);
  }
  if (!supplied.has(SCHEDULER_CACHE_PATH)) {
    log("Legacy bundle has no scheduler config · downloading the pinned scheduler during this explicit import.");
    const response = await fetchModelResource(SCHEDULER_URL, {
      onCacheMiss: path => log("MODEL CACHE MISS · " + path + " · completing legacy bundle import.")
    });
    if (!response.ok || !(await validSchedulerResponse(response))) {
      throw new Error("Could not complete the imported bundle's scheduler config.");
    }
    done++;
  }
  log("Model bundle import complete · " + done + " verified files.");
  return done;
}
