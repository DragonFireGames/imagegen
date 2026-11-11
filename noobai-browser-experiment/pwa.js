if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("./sw.js", { scope: "./", updateViaCache: "none" })
    .catch((error) => console.warn("NoobAI app-shell caching is unavailable:", error));
}
