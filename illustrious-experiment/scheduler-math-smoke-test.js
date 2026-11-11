const BASE_SCHEDULER_CONFIG =
  "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/b41f1c324e32e03f9e104bd9ace0992e4e135597/scheduler/scheduler_config.json";

const button = document.querySelector("#testSchedulerMath");
const status = document.querySelector("#schedulerMathStatus");
const results = document.querySelector("#schedulerMathSummary");
const stepControl = document.querySelector("#schedulerSteps");

function setStatus(message, type = "info") {
  status.textContent = message;
  status.dataset.type = type;
}

function addRow(label, value) {
  const row = document.createElement("div");
  row.className = "data-row";
  const labelNode = document.createElement("span");
  labelNode.className = "data-label";
  labelNode.textContent = label;
  const valueNode = document.createElement("span");
  valueNode.className = "data-value";
  valueNode.textContent = value;
  row.append(labelNode, valueNode);
  results.append(row);
}

function createRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function createNormalNoise(length, seed) {
  const random = createRng(seed);
  const output = new Float32Array(length);
  for (let i = 0; i < length; i += 2) {
    const u1 = Math.max(random(), 1e-9);
    const u2 = random();
    const radius = Math.sqrt(-2 * Math.log(u1));
    output[i] = radius * Math.cos(2 * Math.PI * u2);
    if (i + 1 < length) output[i + 1] = radius * Math.sin(2 * Math.PI * u2);
  }
  return output;
}

function createEulerAncestralScheduler(config, steps) {
  const trainSteps = config.num_train_timesteps;
  const betaStart = config.beta_start;
  const betaEnd = config.beta_end;
  if (config.beta_schedule !== "scaled_linear") {
    throw new Error("Unsupported Illustrious beta schedule: " + config.beta_schedule);
  }
  if (config.prediction_type !== "epsilon") {
    throw new Error("This test currently expects epsilon prediction, not " + config.prediction_type + ".");
  }
  if (config.timestep_spacing !== "leading") {
    throw new Error("Unexpected base timestep_spacing: " + config.timestep_spacing);
  }
  if (config.rescale_betas_zero_snr) {
    throw new Error("This test does not implement rescale_betas_zero_snr.");
  }
  if (config.use_karras_sigmas) {
    throw new Error("This test expects the base configuration's non-Karras sigma schedule.");
  }
  if (config.final_sigmas_type && config.final_sigmas_type !== "zero") {
    throw new Error("Unexpected final sigma policy: " + config.final_sigmas_type);
  }

  const betas = new Float32Array(trainSteps);
  const alphasCumprod = new Float32Array(trainSteps);
  const start = Math.sqrt(betaStart);
  const end = Math.sqrt(betaEnd);
  let cumulative = 1;

  for (let i = 0; i < trainSteps; i++) {
    const rootBeta = Math.fround(start + (end - start) * (i / (trainSteps - 1)));
    const beta = Math.fround(rootBeta * rootBeta);
    betas[i] = beta;
    cumulative = Math.fround(cumulative * Math.fround(1 - beta));
    alphasCumprod[i] = cumulative;
  }

  const trainingSigmas = new Float32Array(trainSteps);
  for (let i = 0; i < trainSteps; i++) {
    trainingSigmas[i] = Math.fround(Math.sqrt((1 - alphasCumprod[i]) / alphasCumprod[i]));
  }

  const ratio = Math.floor(trainSteps / steps);
  const offset = Number(config.steps_offset || 0);
  const timesteps = new Float32Array(steps);
  const sigmas = new Float32Array(steps + 1);
  for (let i = 0; i < steps; i++) {
    const t = (steps - 1 - i) * ratio + offset;
    if (t < 0 || t >= trainSteps) throw new Error("Generated timestep is outside the training schedule.");
    timesteps[i] = t;
    sigmas[i] = trainingSigmas[t];
  }
  sigmas[steps] = 0;

  return {
    timesteps,
    sigmas,
    initNoiseSigma: Math.sqrt(sigmas[0] * sigmas[0] + 1),
    scaleModelInput(sample, index) {
      const sigma = sigmas[index];
      const scale = 1 / Math.sqrt(sigma * sigma + 1);
      const output = new Float32Array(sample.length);
      for (let i = 0; i < sample.length; i++) output[i] = sample[i] * scale;
      return output;
    },
    step(modelOutput, index, sample, noise) {
      const sigma = sigmas[index];
      const sigmaTo = sigmas[index + 1];
      const sigmaUp = Math.sqrt(Math.max(0, sigmaTo * sigmaTo *
        (sigma * sigma - sigmaTo * sigmaTo) / (sigma * sigma)));
      const sigmaDown = Math.sqrt(Math.max(0, sigmaTo * sigmaTo - sigmaUp * sigmaUp));
      const derivative = new Float32Array(sample.length);
      const output = new Float32Array(sample.length);

      for (let i = 0; i < sample.length; i++) {
        const predictedOriginal = sample[i] - sigma * modelOutput[i];
        derivative[i] = (sample[i] - predictedOriginal) / sigma;
        output[i] = sample[i] + derivative[i] * (sigmaDown - sigma) + noise[i] * sigmaUp;
      }

      return { sample: output, sigmaUp, sigmaDown };
    }
  };
}

function runScheduler(config, steps, seed) {
  const scheduler = createEulerAncestralScheduler(config, steps);
  const latentSize = 1024;
  const sample = createNormalNoise(latentSize, seed);
  for (let i = 0; i < sample.length; i++) sample[i] *= scheduler.initNoiseSigma;

  let current = sample;
  let totalNoiseMagnitude = 0;
  let sigmaRelationError = 0;

  for (let step = 0; step < steps; step++) {
    const scaled = scheduler.scaleModelInput(current, step);
    const modelOutput = new Float32Array(latentSize);
    // A deterministic synthetic epsilon prediction; this deliberately does
    // not pretend to be an ONNX UNet prediction.
    for (let i = 0; i < latentSize; i++) {
      modelOutput[i] = Math.sin(i * 0.013 + step * 0.17) * 0.1;
    }
    const noise = createNormalNoise(latentSize, seed + step + 1);
    const result = scheduler.step(modelOutput, step, current, noise);
    const relation = Math.abs(
      result.sigmaUp * result.sigmaUp + result.sigmaDown * result.sigmaDown -
      scheduler.sigmas[step + 1] * scheduler.sigmas[step + 1]
    );
    sigmaRelationError = Math.max(sigmaRelationError, relation);
    totalNoiseMagnitude += result.sigmaUp;
    current = result.sample;
  }

  for (let i = 0; i < current.length; i++) {
    if (!Number.isFinite(current[i])) {
      throw new Error("Scheduler produced a non-finite latent at value " + i + ".");
    }
  }

  return {
    scheduler,
    output: current,
    totalNoiseMagnitude,
    sigmaRelationError
  };
}

button.addEventListener("click", async () => {
  button.disabled = true;
  results.replaceChildren();
  results.hidden = false;

  try {
    const steps = Number(stepControl.value);
    if (!Number.isInteger(steps) || steps < 1 || steps > 100) {
      throw new Error("Choose an inference-step count from 1 to 100.");
    }

    setStatus("Fetching the pinned Illustrious v0.1 scheduler configuration…");
    const response = await fetch(BASE_SCHEDULER_CONFIG, { credentials: "omit" });
    if (!response.ok) throw new Error("Scheduler config request failed: HTTP " + response.status);
    const config = await response.json();

    addRow("Base scheduler config", config._class_name + " · " + (config._diffusers_version || "version unspecified"));
    addRow("Sampler update", "EulerAncestralDiscreteScheduler (euler_a)");
    addRow("Beta schedule", config.beta_schedule + " · " + config.beta_start + " → " + config.beta_end);
    addRow("Prediction type", config.prediction_type);
    addRow("Timestep spacing", config.timestep_spacing + " · offset " + config.steps_offset);
    addRow("Sampler math", "Euler ancestral (euler_a)");
    addRow("Inference steps", String(steps));

    setStatus("Running deterministic 1,024-value scheduler tests (no model weights or GPU sessions)…");
    const first = runScheduler(config, steps, 1337);
    const second = runScheduler(config, steps, 1337);

    let maxRepeatDifference = 0;
    for (let i = 0; i < first.output.length; i++) {
      maxRepeatDifference = Math.max(maxRepeatDifference, Math.abs(first.output[i] - second.output[i]));
    }

    let monotonic = true;
    for (let i = 1; i < first.scheduler.sigmas.length; i++) {
      if (first.scheduler.sigmas[i] > first.scheduler.sigmas[i - 1]) monotonic = false;
    }
    if (!monotonic) throw new Error("Sigma schedule is not monotonically non-increasing.");
    if (maxRepeatDifference !== 0) throw new Error("Seeded scheduler output is not deterministic.");
    if (first.sigmaRelationError > 1e-5) {
      throw new Error("Euler ancestral sigma-up/down identity failed: " + first.sigmaRelationError);
    }

    addRow("Timestep range", first.scheduler.timesteps[0] + " → " +
      first.scheduler.timesteps[first.scheduler.timesteps.length - 1]);
    addRow("Sigma schedule", first.scheduler.sigmas.length + " values · monotonically non-increasing");
    addRow("Initial noise sigma", first.scheduler.initNoiseSigma.toFixed(6));
    addRow("Sigma-up/down identity max error", first.sigmaRelationError.toExponential(2));
    addRow("Deterministic repeat max difference", maxRepeatDifference.toFixed(8));
    addRow("Final synthetic latent", "1 × 1,024 · all finite");
    addRow("Total ancestral noise contribution", first.totalNoiseMagnitude.toFixed(6));

    setStatus("Euler ancestral scheduler math passed for " + steps +
      " deterministic synthetic steps. This validates the scheduler equations and seeded noise only; no UNet was run and no image was generated.", "success");
  } catch (error) {
    console.error("[Illustrious] Scheduler math test failed:", error);
    setStatus("Scheduler math test failed: " + (error?.message || String(error)), "error");
  } finally {
    button.disabled = false;
  }
});
