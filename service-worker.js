const APP_CACHE_NAME = "illustrious-app-shell-source-v2";
const APP_SCOPE = new URL("./illustrious-experiment/", self.registration.scope);
const SHELL_ASSETS = [
  "./",
  "./index.html",
  "./text-to-image.js",
  "./model-cache.js",
  "./lora-patcher.js",
  "./heavy-model-test-lock.js",
  "../vendor/transformers/transformers.min.js",
  "../vendor/onnxruntime-web/ort.jspi.bundle.min.mjs",
  "../vendor/onnxruntime-web/ort-wasm-simd-threaded.jspi.mjs",
  "../vendor/onnxruntime-web/ort-wasm-simd-threaded.jspi.wasm",
  "../vendor/onnxruntime-web/ort.webgpu.min.mjs",
  "../vendor/onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs",
  "../vendor/onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm"
];
const SHELL_URLS = new Set(SHELL_ASSETS.map(path => new URL(path, APP_SCOPE).href));
const OPTIONAL_ASSETS = new Set([
  "../vendor/onnxruntime-web/ort.jspi.bundle.min.mjs",
  "../vendor/onnxruntime-web/ort-wasm-simd-threaded.jspi.mjs",
  "../vendor/onnxruntime-web/ort-wasm-simd-threaded.jspi.wasm"
].map(path => new URL(path, APP_SCOPE).href));

self.addEventListener("install", event => {
  event.waitUntil((async () => {
    const cache = await caches.open(APP_CACHE_NAME);
    for (const asset of SHELL_ASSETS) {
      const url = new URL(asset, APP_SCOPE);
      try {
        const response = await fetch(url, { cache: "reload" });
        if (!response.ok) throw new Error("HTTP " + response.status);
        await cache.put(url.href, response);
      } catch (error) {
        if (OPTIONAL_ASSETS.has(url.href)) {
          console.warn("[Illustrious service worker] Optional JSPI asset is not available yet: " + url.pathname);
          continue;
        }
        throw new Error("Could not cache required offline app asset " + url.pathname + ": " + (error?.message || error));
      }
    }
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", event => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names
      .filter(name => name.startsWith("illustrious-app-shell-") && name !== APP_CACHE_NAME)
      .map(name => caches.delete(name)));
    await self.clients.claim();
  })());
});

function isRefreshableAppSource(url) {
  return /\\.(?:html|js|mjs|css)$/.test(url.pathname);
}

self.addEventListener("fetch", event => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  // The worker has project-root scope to control ../vendor assets, but only
  // handles this experiment and its shared vendor directory.
  const inExperiment = url.href.startsWith(APP_SCOPE.href);
  const vendorScope = new URL("../vendor/", APP_SCOPE).href;
  const inVendor = url.href.startsWith(vendorScope);
  if (!inExperiment && !inVendor) return;

  if (request.mode === "navigate" && inExperiment) {
    event.respondWith((async () => {
      try {
        const response = await fetch(request);
        if (response.ok) {
          const cache = await caches.open(APP_CACHE_NAME);
          await cache.put(request, response.clone());
        }
        return response;
      } catch {
        const cached = await caches.match(request);
        return cached || Response.error();
      }
    })());
    return;
  }

  if (!SHELL_URLS.has(url.href)) return;
  event.respondWith((async () => {
    const cache = await caches.open(APP_CACHE_NAME);
    const cached = await cache.match(request);

    // Revalidate app source on every load so a deployed module cannot remain
    // pinned to an older Cache Storage entry. Keep the last good copy for offline use.
    if (isRefreshableAppSource(url)) {
      try {
        const response = await fetch(new Request(request, { cache: "no-cache" }));
        if (response.ok) {
          await cache.put(request, response.clone());
          return response;
        }
        if (cached) return cached;
        return response;
      } catch (error) {
        if (cached) return cached;
        throw error;
      }
    }

    if (cached) return cached;
    const response = await fetch(request);
    if (response.ok) await cache.put(request, response.clone());
    return response;
  })());
});
