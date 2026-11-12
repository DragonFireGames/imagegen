export const onnxSession = {
    sessions: {},
    backend: "auto",
    async getSession(onnx_path) {
        if (!(onnx_path in this.sessions)) {
            let ep;
            if (this.backend === "webgpu") {
                ep = ["webgpu"];
            } else if (this.backend === "wasm") {
                ep = ["wasm"];
            } else {
                ep = ["webgpu", "wasm"];
            }
            try {
                this.sessions[onnx_path] = await ort.InferenceSession.create(
                    onnx_path,
                    // webgl provider does not work due to various problems
                    {
                        logSeverityLevel: 3,
                        executionProviders: ep,
                    }
                );
            } catch (error) {
                delete this.sessions[onnx_path];
                console.error("ONNX model load failed:", onnx_path, error);
                throw new Error(
                    `Could not load ONNX model: ${onnx_path}. Check that the file exists and that its host allows cross-origin requests (CORS). ${error.message || error}`
                );
            }
        }
        return this.sessions[onnx_path];
    }
};

export function decodeImage(image) {
    if (!image || !image.complete || image.naturalWidth <= 0 || image.naturalHeight <= 0) {
        throw new Error("The selected image is not fully loaded or has invalid dimensions.");
    }

    const width = image.naturalWidth;
    const height = image.naturalHeight;
    const canvas = typeof OffscreenCanvas !== "undefined"
        ? new OffscreenCanvas(width, height)
        : Object.assign(document.createElement("canvas"), { width, height });
    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    if (!ctx) {
        throw new Error("Could not create a 2D canvas context to decode the selected image.");
    }

    ctx.drawImage(image, 0, 0, width, height);
    return ctx.getImageData(0, 0, width, height);
}

export function checkClipboard() {
    return ("clipboard" in navigator) && ("read" in navigator.clipboard);
}

export async function readFromClipboard() {
    try {
        const items = await navigator.clipboard.read();
        for (const item of items) {
            const mime = item.types.find(type => type.startsWith("image/"));
            if (mime) {
                const blob = await item.getType(mime);
                return blob;
            }
        }
    } catch (_e) {
        console.error(_e);
    }
    return null;
}

export function uuid() {
    // ref: http://stackoverflow.com/a/2117523
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0, v = c == "x" ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

export function debugPrintImageData(image_data) {
    const canvas = document.createElement("canvas");
    canvas.width = image_data.width;
    canvas.height = image_data.height;
    const ctx = canvas.getContext("2d");
    ctx.putImageData(image_data, 0, 0);
    document.body.append(canvas);
}
