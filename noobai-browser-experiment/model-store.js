const CACHE_DIRECTORY = "noobai-xl-1.1-browser-cache-v1";

function joinUrl(base, path) {
  return new URL(path.replace(/^\/+/, ""), base.endsWith("/") ? base : base + "/").toString();
}

async function rootDirectory() {
  if (!navigator.storage?.getDirectory) {
    throw new Error("This browser does not expose the Origin Private File System (OPFS). Use a current Chromium browser.");
  }
  return navigator.storage.getDirectory();
}

async function cacheDirectory(key) {
  const root = await rootDirectory();
  const cacheRoot = await root.getDirectoryHandle(CACHE_DIRECTORY, { create: true });
  return cacheRoot.getDirectoryHandle(key, { create: true });
}

async function urlKey(url) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(url));
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
}

async function downloadFile(url, expectedBytes, onProgress, signal, cacheIdentity = url) {
  const directory = await cacheDirectory(await urlKey(cacheIdentity));
  const filename = "asset.bin";
  const handle = await directory.getFileHandle(filename, { create: true });
  let existing = await handle.getFile();

  if (expectedBytes > 0 && existing.size === expectedBytes) {
    onProgress?.({ cached: true, loaded: expectedBytes, total: expectedBytes });
    return existing;
  }

  if (expectedBytes > 0 && existing.size > expectedBytes) {
    const reset = await handle.createWritable({ keepExistingData: false });
    await reset.close();
    existing = await handle.getFile();
  }

  let offset = existing.size;
  const headers = {};
  if (offset > 0) headers.Range = "bytes=" + offset + "-";

  let response = await fetch(url, { headers, cache: "no-store", signal });
  if (offset > 0 && response.status === 200) {
    // The host ignored Range; restart instead of appending a duplicate file.
    offset = 0;
    const reset = await handle.createWritable({ keepExistingData: false });
    await reset.close();
    response = await fetch(url, { cache: "no-store", signal });
  }
  if (offset > 0 && response.status !== 206) {
    throw new Error("The model host did not honor a resume request for " + url + " (HTTP " + response.status + ").");
  }
  if (offset === 0 && !response.ok) {
    throw new Error("Could not download " + url + " (HTTP " + response.status + ").");
  }

  const total = expectedBytes || (() => {
    const range = response.headers.get("Content-Range");
    const match = range && range.match(/\/(\d+)$/);
    if (match) return Number(match[1]);
    const length = Number(response.headers.get("Content-Length"));
    return Number.isFinite(length) && length > 0 ? offset + length : 0;
  })();

  const writer = await handle.createWritable({ keepExistingData: true });
  await writer.seek(offset);
  let loaded = offset;
  try {
    if (!response.body) throw new Error("The model host returned no readable response body.");
    const reader = response.body.getReader();
    while (true) {
      if (signal?.aborted) throw new Error("Download cancelled.");
      const { value, done } = await reader.read();
      if (done) break;
      await writer.write(value);
      loaded += value.byteLength;
      onProgress?.({ cached: false, loaded, total });
    }
    await writer.close();
  } catch (error) {
    // Commit the received prefix so a later visit can resume with an HTTP Range.
    await writer.close().catch(() => {});
    throw error;
  }

  const file = await handle.getFile();
  if (expectedBytes > 0 && file.size !== expectedBytes) {
    throw new Error("Incomplete model file: expected " + expectedBytes + " bytes, received " + file.size + ". Revisit to resume the download.");
  }
  return file;
}

export async function loadModelPack(baseUrl, onProgress, signal) {
  const normalizedBase = baseUrl.replace(/\/+$/, "") + "/";
  const manifestUrl = joinUrl(normalizedBase, "manifest.json");
  const manifestDirectory = await cacheDirectory(await urlKey("manifest:" + normalizedBase));
  const manifestHandle = await manifestDirectory.getFileHandle("manifest.json", { create: true });
  let manifest;
  try {
    const manifestResponse = await fetch(manifestUrl, { cache: "no-store", signal });
    if (!manifestResponse.ok) {
      throw new Error("Could not read model manifest " + manifestUrl + " (HTTP " + manifestResponse.status + ").");
    }
    manifest = await manifestResponse.json();
    const writer = await manifestHandle.createWritable();
    await writer.write(JSON.stringify(manifest));
    await writer.close();
  } catch (error) {
    if (signal?.aborted) throw error;
    const cachedManifest = await manifestHandle.getFile();
    if (!cachedManifest.size) throw error;
    manifest = JSON.parse(await cachedManifest.text());
    onProgress?.({
      phase: "cache",
      file: "manifest.json",
      loaded: cachedManifest.size,
      total: cachedManifest.size,
    });
  }
  if (manifest.format !== "noobai-sdxl-browser-onnx-v1" || !manifest.components) {
    throw new Error("This is not a supported NoobAI browser ONNX pack (manifest format mismatch).");
  }
  const packVersion = manifest.packId || manifest.sourceRevision || "unversioned";

  const assets = [];
  for (const component of Object.values(manifest.components)) {
    assets.push({
      path: component.graphBytesAsset,
      bytes: component.graphBytes,
    });
    for (const item of component.externalData || []) {
      assets.push({ path: item.asset, bytes: item.bytes });
    }
  }
  const unique = new Map(assets.map((item) => [item.path, item]));
  const files = Array.from(unique.values());
  const totalBytes = files.reduce((sum, item) => sum + (item.bytes || 0), 0);
  let completedBytes = 0;

  for (const asset of files) {
    if (signal?.aborted) throw new Error("Model download cancelled.");
    const url = joinUrl(normalizedBase, asset.path);
    const cacheIdentity = normalizedBase + "|" + packVersion + "|" + asset.path;
    const file = await downloadFile(url, asset.bytes || 0, (progress) => {
      const overall = completedBytes + progress.loaded;
      onProgress?.({
        phase: progress.cached ? "cache" : "download",
        file: asset.path,
        loaded: overall,
        total: totalBytes || overall,
        fileLoaded: progress.loaded,
        fileTotal: progress.total,
      });
    }, signal, cacheIdentity);
    completedBytes += file.size;
  }

  return {
    baseUrl: normalizedBase,
    manifest,
    totalBytes,
    cachedBytes: completedBytes,
  };
}

export async function getCachedModelFile(baseUrl, assetPath, packVersion = "unversioned") {
  const normalizedBase = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
  const url = joinUrl(normalizedBase, assetPath);
  const cacheIdentity = normalizedBase + "|" + packVersion + "|" + assetPath;
  const directory = await cacheDirectory(await urlKey(cacheIdentity));
  const handle = await directory.getFileHandle("asset.bin");
  return handle.getFile();
}

export async function clearModelCache() {
  const root = await rootDirectory();
  await root.removeEntry(CACHE_DIRECTORY, { recursive: true }).catch((error) => {
    if (error.name !== "NotFoundError") throw error;
  });
}
