const $ = (id) => document.getElementById(id);

const form = $("generation-form");
const generateButton = $("generate");
const connectionDot = $("connection-dot");
const connectionLabel = $("connection-label");
const setupTitle = $("setup-title");
const setupDetail = $("setup-detail");
const progressLabel = $("progress-label");
const progressExtra = $("progress-extra");
const progressBar = $("progress-bar");
const rendered = $("rendered");
const placeholder = $("placeholder");
const download = $("download");
const log = $("log");
let busy = false;

function setProgress(label, percent, extra = "") {
  progressLabel.textContent = label;
  progressExtra.textContent = extra;
  progressBar.style.width = Math.max(0, Math.min(100, percent)) + "%";
}

async function getJson(url, options) {
  const response = await fetch(url, { cache: "no-store", ...options });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || "Request failed (" + response.status + ").");
  return body;
}

async function checkEngine() {
  try {
    const status = await getJson("/api/status");
    const missing = status.missing || [];
    if (status.ready) {
      connectionDot.classList.add("ready");
      connectionLabel.textContent = "Local engine ready";
      setupTitle.textContent = "Pony engine is ready";
      setupDetail.textContent = "The sd-cli executable and Pony Diffusion model weights were found. Generation is processed on this machine.";
      generateButton.disabled = false;
      generateButton.textContent = "Generate locally";
      $("result-info").textContent = "Ready · " + (status.offloadToCpu ? "CPU offload enabled" : "CPU offload disabled");
    } else {
      connectionDot.classList.add("error");
      connectionLabel.textContent = "Setup required";
      setupTitle.textContent = "Local setup is incomplete";
      setupDetail.textContent = missing.length ? "Missing: " + missing.join(", ") + ". Open the setup guide for the macOS Metal build and model downloads." : (status.warning || "Check the local configuration.");
      generateButton.disabled = true;
      generateButton.textContent = "Complete local setup first";
    }
  } catch (error) {
    connectionDot.classList.add("error");
    connectionLabel.textContent = "Local server not running";
    setupTitle.textContent = "Start the local server";
    setupDetail.textContent = "This GitHub Pages page cannot run native C++ inference. From the repository root, run: node uncensored-experiment/server.mjs — then open http://127.0.0.1:8788.";
    generateButton.disabled = true;
    generateButton.textContent = "Start the local server";
    $("result-info").textContent = "The native backend is not reachable from this page.";
    log.textContent = error.message;
  }
}

function setBusy(value) {
  busy = value;
  generateButton.disabled = value || connectionLabel.textContent !== "Local engine ready";
  generateButton.textContent = value ? "Generating…" : (generateButton.disabled ? "Local engine unavailable" : "Generate locally");
}

async function pollJob(id) {
  while (true) {
    const job = await getJson("/api/jobs/" + encodeURIComponent(id));
    log.textContent = job.log || "";
    log.scrollTop = log.scrollHeight;
    if (job.status === "complete") {
      rendered.src = job.imageUrl + "?t=" + Date.now();
      rendered.hidden = false;
      placeholder.hidden = true;
      download.href = job.imageUrl;
      download.hidden = false;
      $("seed-info").textContent = "Seed " + job.seed;
      $("result-info").textContent = job.elapsedSeconds.toFixed(1) + "s · " + $("width").value + " × " + $("height").value;
      setProgress("Complete", 100, "Seed " + job.seed);
      return;
    }
    if (job.status === "failed") throw new Error(job.error || "Generation failed. See the log.");
    setProgress("Generating on this device", 35, "Seed " + job.seed);
    await new Promise((resolve) => setTimeout(resolve, 1200));
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;

  const payload = {
    prompt: $("prompt").value,
    negativePrompt: $("negative-prompt").value,
    width: Number($("width").value),
    height: Number($("height").value),
    steps: Number($("steps").value),
    cfgScale: Number($("cfg").value),
    seed: Number($("seed").value)
  };

  if (!payload.prompt.trim()) {
    $("prompt").focus();
    return;
  }

  setBusy(true);
  download.hidden = true;
  rendered.hidden = true;
  placeholder.hidden = false;
  $("seed-info").textContent = "";
  $("result-info").textContent = "Preparing local inference";
  setProgress("Submitting generation", 5);

  try {
    const created = await getJson("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    $("seed").value = created.seed;
    await pollJob(created.id);
  } catch (error) {
    $("result-info").textContent = error.message;
    $("result-info").classList.add("error");
    setProgress("Generation failed", 0);
    log.textContent += "\n" + error.message;
  } finally {
    $("result-info").classList.remove("error");
    setBusy(false);
  }
});

$("cfg").addEventListener("input", () => {
  $("cfg-value").textContent = Number($("cfg").value).toFixed(1);
});

for (const preset of document.querySelectorAll(".preset")) {
  preset.addEventListener("click", () => {
    const size = preset.dataset.size;
    $("width").value = preset.dataset.width || size;
    $("height").value = preset.dataset.height || size;
    document.querySelectorAll(".preset").forEach((button) => button.classList.toggle("active", button === preset));
  });
}
for (const field of [$("width"), $("height")]) {
  field.addEventListener("change", () => {
    document.querySelectorAll(".preset").forEach((button) => {
      const width = Number(button.dataset.width || button.dataset.size);
      const height = Number(button.dataset.height || button.dataset.size);
      button.classList.toggle("active", width === Number($("width").value) && height === Number($("height").value));
    });
  });
}

checkEngine();
