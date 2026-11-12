import { AppUI } from "./ui.js?v=20261112-fix2";

document.addEventListener("DOMContentLoaded", () => {
    // onnxruntime-web configuration
    if (typeof ort !== "undefined") {
        ort.env.wasm.proxy = true;
        // Multithreaded WASM requires cross-origin isolation (COOP + COEP).
        // GitHub Pages is not isolated here, so avoid a guaranteed worker fallback.
        ort.env.wasm.numThreads = window.crossOriginIsolated
            ? (navigator.hardwareConcurrency || 4)
            : 1;
    }

    // Initialize UI
    new AppUI();
});
