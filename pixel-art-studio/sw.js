const CACHE_NAME = "pixel-foundry-runtime-v1";
const CDN_HOSTS = new Set(["cdn.jsdelivr.net"]);

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((key) => key.startsWith("pixel-foundry-runtime-") && key !== CACHE_NAME).map((key) => caches.delete(key)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.protocol !== "https:" || !CDN_HOSTS.has(url.hostname)) return;

  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(request, { ignoreVary: true });
    if (cached) return cached;

    const response = await fetch(request);
    if (response.ok || response.type === "opaque") {
      try {
        await cache.put(request, response.clone());
      } catch (error) {
        console.warn("Could not cache runtime asset for offline use:", request.url, error);
      }
    }
    return response;
  })());
});
