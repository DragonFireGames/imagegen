const CACHE_NAME = "noobai-browser-app-shell-v3";
const APP_ASSETS = [
  "./",
  "./index.html",
  "./styles.css",
  "./app.js",
  "./worker.js",
  "./model-store.js",
  "./pwa.js",
  "./README.md",
  "./vendor/ort.jspi.bundle.min.mjs",
  "./vendor/ort-wasm-simd-threaded.jspi.mjs",
  "./vendor/ort-wasm-simd-threaded.jspi.wasm",
  "./vendor/transformers/transformers.min.js",
];

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE_NAME);
    await Promise.allSettled(APP_ASSETS.map((asset) => cache.add(asset)));
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names.filter((name) => name.startsWith("noobai-browser-app-shell-") && name !== CACHE_NAME)
      .map((name) => caches.delete(name)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  const path = url.pathname;
  const appPath = new URL("./", self.registration.scope).pathname;
  const isExperimentAsset = path.startsWith(appPath);
  const isSharedTokenizerAsset = path.includes("/vendor/transformers/");
  if (!isExperimentAsset && !isSharedTokenizerAsset) return;

  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(request);
    if (cached) return cached;
    const response = await fetch(request);
    if (response.ok && response.type === "basic") {
      cache.put(request, response.clone()).catch(() => {});
    }
    return response;
  })());
});
