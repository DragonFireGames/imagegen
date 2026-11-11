if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("./sw.js", { scope: "./" })
    .catch((error) => console.warn("NoobAI app-shell caching is unavailable:", error));
}
