import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { basename, dirname, extname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { randomInt, randomUUID } from "node:crypto";

const EXPERIMENT_DIR = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(EXPERIMENT_DIR, "..");
const OUTPUT_DIR = join(EXPERIMENT_DIR, "outputs");
const CONFIG_PATH = join(EXPERIMENT_DIR, "config.local.json");
const MAX_BODY_BYTES = 24 * 1024;
const MAX_LOG_CHARS = 16000;
const MAX_JOBS = 24;

mkdirSync(OUTPUT_DIR, { recursive: true });

let localConfig = {};
let configWarning = "";
if (existsSync(CONFIG_PATH)) {
  try {
    localConfig = JSON.parse(readFileSync(CONFIG_PATH, "utf8"));
  } catch (error) {
    configWarning = "Could not read config.local.json: " + error.message;
  }
}

function configuredPath(configKey, envKey, fallback) {
  const value = process.env[envKey] || localConfig[configKey] || fallback;
  return isAbsolute(value) ? value : resolve(PROJECT_ROOT, value);
}

const paths = {
  cli: configuredPath("sdCli", "SD_CLI", ".local/stable-diffusion.cpp/build/bin/sd-cli"),
  model: configuredPath("model", "PONY_MODEL", ".local/pony-diffusion-v6-xl/pony-diffusion-v6-xl-Q8_0.gguf"),
  offloadToCpu: localConfig.offloadToCpu !== false
};

const jobs = new Map();
let activeJobId = null;

function json(res, status, value) {
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff"
  });
  res.end(JSON.stringify(value));
}

function addLog(job, value) {
  job.log = (job.log + value).slice(-MAX_LOG_CHARS);
}

function getStatus() {
  const inspect = (file) => {
    const exists = existsSync(file);
    let size = 0;
    if (exists) {
      try { size = statSync(file).size; } catch {}
    }
    return { path: file, exists, size };
  };

  const engine = inspect(paths.cli);
  const model = inspect(paths.model);
  const missing = [
    !engine.exists && "stable-diffusion.cpp sd-cli executable",
    !model.exists && "Pony Diffusion V6 XL model weights"
  ].filter(Boolean);

  return {
    ready: missing.length === 0 && !configWarning,
    engine,
    model,
    offloadToCpu: paths.offloadToCpu,
    missing,
    warning: configWarning
  };
}

async function readJsonBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) throw new Error("Request body is too large.");
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function validateGeneration(body) {
  const prompt = typeof body.prompt === "string" ? body.prompt.trim() : "";
  const negativePrompt = typeof body.negativePrompt === "string" ? body.negativePrompt.trim() : "";
  const width = Number(body.width);
  const height = Number(body.height);
  const steps = Number(body.steps);
  const cfgScale = Number(body.cfgScale);
  const requestedSeed = Number(body.seed);

  if (!prompt || prompt.length > 6000) throw new Error("Enter a prompt (maximum 6,000 characters).");
  if (negativePrompt.length > 4000) throw new Error("Negative prompt must be 4,000 characters or fewer.");
  if (!Number.isInteger(width) || !Number.isInteger(height) ||
      width < 512 || height < 512 || width > 1536 || height > 1536 ||
      width % 64 !== 0 || height % 64 !== 0) {
    throw new Error("Width and height must be 512–1536 pixels and divisible by 64.");
  }
  if (!Number.isInteger(steps) || steps < 10 || steps > 60) {
    throw new Error("Steps must be an integer from 10 to 60.");
  }
  if (!Number.isFinite(cfgScale) || cfgScale < 1 || cfgScale > 12) {
    throw new Error("Guidance scale must be between 1 and 12.");
  }
  if (!Number.isInteger(requestedSeed) || requestedSeed < -1 || requestedSeed > 2147483647) {
    throw new Error("Seed must be -1 (random) or an integer from 0 to 2147483647.");
  }

  return {
    prompt,
    negativePrompt,
    width,
    height,
    steps,
    cfgScale,
    seed: requestedSeed < 0 ? randomInt(0, 2147483647) : requestedSeed
  };
}

function startGeneration(input) {
  const id = randomUUID();
  const outputName = "pony-" + id + ".png";
  const outputPath = join(OUTPUT_DIR, outputName);
  const job = {
    id,
    status: "running",
    seed: input.seed,
    startedAt: Date.now(),
    elapsedSeconds: null,
    imageUrl: null,
    error: null,
    log: ""
  };
  jobs.set(id, job);
  activeJobId = id;

  const args = [
    "--model", paths.model,
    "--prompt", input.prompt,
    "--negative-prompt", input.negativePrompt,
    "--width", String(input.width),
    "--height", String(input.height),
    "--steps", String(input.steps),
    "--cfg-scale", String(input.cfgScale),
    "--sampling-method", "euler_a",
    "--clip-skip", "2",
    "--seed", String(input.seed),
    "--output", outputPath,
    "--log-level", "verbose",
    "--vae-tiling"
  ];
  if (paths.offloadToCpu) args.push("--offload-to-cpu");

  addLog(job, "Starting local Pony Diffusion V6 XL inference. Seed " + input.seed + ".\n");
  const child = spawn(paths.cli, args, {
    cwd: PROJECT_ROOT,
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true
  });

  child.stdout.on("data", (chunk) => addLog(job, chunk.toString()));
  child.stderr.on("data", (chunk) => addLog(job, chunk.toString()));
  child.on("error", (error) => {
    job.status = "failed";
    job.error = "Could not start sd-cli: " + error.message;
    addLog(job, "\n" + job.error + "\n");
    job.elapsedSeconds = (Date.now() - job.startedAt) / 1000;
    activeJobId = null;
  });
  child.on("close", (code, signal) => {
    job.elapsedSeconds = (Date.now() - job.startedAt) / 1000;
    if (job.status !== "running") return;
    if (code === 0 && existsSync(outputPath)) {
      job.status = "complete";
      job.imageUrl = "/outputs/" + outputName;
      addLog(job, "\nGeneration completed in " + job.elapsedSeconds.toFixed(1) + " seconds.\n");
    } else {
      job.status = "failed";
      job.error = "sd-cli exited with code " + code + (signal ? " (signal " + signal + ")" : "") + ". See the log for details.";
      addLog(job, "\n" + job.error + "\n");
    }
    activeJobId = null;
  });

  while (jobs.size > MAX_JOBS) {
    const firstId = jobs.keys().next().value;
    if (firstId === activeJobId) break;
    jobs.delete(firstId);
  }

  return job;
}

function sendStatic(res, filePath) {
  const ext = extname(filePath).toLowerCase();
  const contentType = ext === ".html" ? "text/html; charset=utf-8"
    : ext === ".css" ? "text/css; charset=utf-8"
    : ext === ".js" ? "text/javascript; charset=utf-8"
    : ext === ".md" ? "text/markdown; charset=utf-8"
    : ext === ".png" ? "image/png"
    : "application/octet-stream";
  res.writeHead(200, {
    "Content-Type": contentType,
    "Cache-Control": ext === ".png" ? "no-store" : "no-cache",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
  });
  res.end(readFileSync(filePath));
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://127.0.0.1");
    const pathname = decodeURIComponent(url.pathname);

    if (req.method === "GET" && pathname === "/api/status") {
      json(res, 200, getStatus());
      return;
    }

    if (req.method === "POST" && pathname === "/api/generate") {
      if (activeJobId && jobs.get(activeJobId)?.status === "running") {
        json(res, 409, { error: "A generation is already running.", jobId: activeJobId });
        return;
      }
      let input;
      try {
        input = validateGeneration(await readJsonBody(req));
      } catch (error) {
        json(res, 400, { error: error.message });
        return;
      }

      const status = getStatus();
      if (!status.ready) {
        json(res, 503, { error: "The local engine is not ready.", status });
        return;
      }

      const job = startGeneration(input);
      json(res, 202, { id: job.id, seed: job.seed, status: job.status });
      return;
    }

    const jobMatch = pathname.match(/^\/api\/jobs\/([0-9a-f-]{36})$/i);
    if (req.method === "GET" && jobMatch) {
      const job = jobs.get(jobMatch[1]);
      if (!job) {
        json(res, 404, { error: "Job not found." });
        return;
      }
      json(res, 200, {
        id: job.id,
        status: job.status,
        seed: job.seed,
        elapsedSeconds: job.elapsedSeconds,
        imageUrl: job.imageUrl,
        error: job.error,
        log: job.log
      });
      return;
    }

    if (req.method === "GET" && pathname.startsWith("/outputs/")) {
      const filename = basename(pathname.slice("/outputs/".length));
      if (!/^pony-[0-9a-f-]{36}\.png$/i.test(filename)) {
        json(res, 404, { error: "File not found." });
        return;
      }
      const file = join(OUTPUT_DIR, filename);
      if (!existsSync(file)) {
        json(res, 404, { error: "File not found." });
        return;
      }
      sendStatic(res, file);
      return;
    }

    if (req.method === "GET" && ["/", "/index.html", "/styles.css", "/app.js", "/README.md"].includes(pathname)) {
      const filename = pathname === "/" || pathname === "/index.html" ? "index.html" : pathname.slice(1);
      sendStatic(res, join(EXPERIMENT_DIR, filename));
      return;
    }

    json(res, 404, { error: "Not found." });
  } catch (error) {
    if (!res.headersSent) json(res, 500, { error: error.message || "Internal server error." });
    else res.destroy();
  }
});

const port = Number(process.env.PORT || 8788);
if (!Number.isInteger(port) || port < 1024 || port > 65535) {
  throw new Error("PORT must be an integer from 1024 to 65535.");
}

server.listen(port, "127.0.0.1", () => {
  console.log("Pony Diffusion V6 XL local experiment: http://127.0.0.1:" + port);
  console.log("Bound to 127.0.0.1 only; requests and model inference stay on this machine.");
  const status = getStatus();
  if (!status.ready) console.log("Setup needed: " + status.missing.join(", "));
});
