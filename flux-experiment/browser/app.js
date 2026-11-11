const $ = (id) => document.getElementById(id);
let lastReport = null;
let adapter = null;
let device = null;

function log(message) {
  const line = "[" + new Date().toLocaleTimeString() + "] " + message;
  const output = $("log");
  output.textContent = (output.textContent ? output.textContent + "\n" : "") + line;
  output.scrollTop = output.scrollHeight;
}

function formatGiB(bytes) {
  return (bytes / 1024 ** 3).toFixed(2) + " GiB";
}

function metric(label, value) {
  return '<div class="metric"><span>' + label + '</span><strong>' + value + '</strong></div>';
}

async function inspectDevice() {
  $("gpu-dot").className = "dot";
  $("gpu-status").textContent = "Checking WebGPU…";
  $("run-benchmark").disabled = true;
  $("device-report").innerHTML = "<p>Requesting a high-performance WebGPU adapter…</p>";
  const report = {
    checkedAt: new Date().toISOString(),
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    crossOriginIsolated: self.crossOriginIsolated,
    secureContext: self.isSecureContext,
    webgpu: Boolean(navigator.gpu),
  };

  if (!navigator.gpu) {
    $("gpu-dot").classList.add("error");
    $("gpu-status").textContent = "WebGPU unavailable";
    $("device-report").innerHTML = "<p>navigator.gpu is unavailable. Open this page on a current Chromium browser over HTTPS.</p>";
    log("ERROR · navigator.gpu is unavailable.");
    lastReport = report;
    return;
  }

  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("requestAdapter returned null.");
    const info = adapter.info || {};
    const limits = adapter.limits;
    const features = Array.from(adapter.features || []);
    report.adapter = {
      vendor: info.vendor || "not exposed",
      architecture: info.architecture || "not exposed",
      device: info.device || "not exposed",
      description: info.description || "not exposed",
      features,
      limits: {
        maxBufferSize: limits.maxBufferSize,
        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
        maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
        maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
      },
    };
    const wantsF16 = adapter.features.has("shader-f16");
    const requiredFeatures = wantsF16 ? ["shader-f16"] : [];
    device = await adapter.requestDevice({ requiredFeatures });
    report.shaderF16 = wantsF16;
    report.deviceLimits = {
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
      maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
    };
    report.deviceFeatures = Array.from(device.features || []);
    $("device-report").innerHTML = '<div class="report-grid">' +
      metric("Adapter", [info.vendor, info.architecture].filter(Boolean).join(" · ") || "Not exposed") +
      metric("Device", info.device || info.description || "Not exposed") +
      metric("FP16 shader support", wantsF16 ? "Yes · shader-f16" : "No") +
      metric("Max buffer", formatGiB(limits.maxBufferSize)) +
      metric("Max storage binding", formatGiB(limits.maxStorageBufferBindingSize)) +
      metric("Compute invocations / workgroup", String(limits.maxComputeInvocationsPerWorkgroup)) +
      metric("Storage buffers / shader stage", String(limits.maxStorageBuffersPerShaderStage)) +
      metric("Cross-origin isolated", String(self.crossOriginIsolated)) +
      '</div><p class="hint">WebGPU does not expose total VRAM or free GPU memory. These are adapter limits, not a VRAM report.</p>';
    $("gpu-dot").classList.add("ready");
    $("gpu-status").textContent = "WebGPU ready";
    $("run-benchmark").disabled = false;
    log("WebGPU adapter ready · " + (info.vendor || "vendor unknown") + " · " +
      (info.architecture || "architecture unknown") + " · maxBufferSize " + formatGiB(limits.maxBufferSize));
    log("FP16 shader support: " + (wantsF16 ? "available" : "not available") +
      " · timestamp-query " + (adapter.features.has("timestamp-query") ? "available" : "not available"));
    if (limits.maxStorageBufferBindingSize < 1024 ** 3) {
      log("WARNING · maxStorageBufferBindingSize is below 1 GiB; full-size transformer tensor placement will need careful sharding.");
    }
    lastReport = report;
  } catch (error) {
    $("gpu-dot").classList.add("error");
    $("gpu-status").textContent = "WebGPU setup failed";
    $("device-report").innerHTML = "<p>" + error.message + "</p>";
    log("ERROR · " + (error.stack || error.message));
    lastReport = { ...report, error: error.message };
  }
}

function makeShader(dtype) {
  const type = dtype === "f16" ? "f16" : "f32";
  const cast = dtype === "f16" ? "f32(A[row * n + k]) * f32(B[k * n + col])" : "A[row * n + k] * B[k * n + col]";
  return (dtype === "f16" ? "enable f16;\n" : "") + `
struct Params { n: u32 };
@group(0) @binding(0) var<storage, read> A: array<${type}>;
@group(0) @binding(1) var<storage, read> B: array<${type}>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let row = gid.y;
  if (row >= params.n || col >= params.n) { return; }
  var sum = 0.0;
  for (var k = 0u; k < params.n; k = k + 1u) {
    sum = sum + ${cast};
  }
  C[row * params.n + col] = sum;
}
`;
}

async function runOneMatmul(n, dtype) {
  const itemBytes = dtype === "f16" ? 2 : 4;
  const count = n * n;
  const aBytes = count * itemBytes;
  const cBytes = count * 4;
  const makeInput = () => {
    const values = new Float32Array(count);
    for (let i = 0; i < count; i++) values[i] = Math.sin(i * 0.017) * 0.5;
    if (dtype === "f16") {
      const half = new Uint16Array(count);
      const f32 = new Float32Array(1);
      const u32 = new Uint32Array(f32.buffer);
      for (let i = 0; i < count; i++) {
        f32[0] = values[i];
        const bits = u32[0];
        const sign = (bits >>> 16) & 0x8000;
        let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
        let mantissa = bits & 0x7fffff;
        if (exponent <= 0) {
          half[i] = exponent < -10 ? sign : sign | (((mantissa | 0x800000) >> (1 - exponent) + 0x1000) >> 13);
        } else if (exponent >= 31) {
          half[i] = sign | 0x7c00;
        } else {
          mantissa += 0x1000 + ((mantissa >> 13) & 1);
          if (mantissa & 0x800000) { mantissa = 0; exponent++; }
          half[i] = exponent >= 31 ? sign | 0x7c00 : sign | (exponent << 10) | (mantissa >> 13);
        }
      }
      return half;
    }
    return values;
  };
  const a = makeInput();
  const b = makeInput();
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const bufferA = device.createBuffer({ size: aBytes, usage });
  const bufferB = device.createBuffer({ size: aBytes, usage });
  const bufferC = device.createBuffer({ size: cBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  device.queue.writeBuffer(bufferA, 0, a);
  device.queue.writeBuffer(bufferB, 0, b);
  device.queue.writeBuffer(params, 0, new Uint32Array([n, 0, 0, 0]));
  const module = device.createShaderModule({ code: makeShader(dtype) });
  const info = await module.getCompilationInfo();
  const shaderErrors = info.messages.filter((message) => message.type === "error");
  if (shaderErrors.length) throw new Error("WGSL compile failed: " + shaderErrors.map((message) => message.message).join("; "));
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufferA } },
      { binding: 1, resource: { buffer: bufferB } },
      { binding: 2, resource: { buffer: bufferC } },
      { binding: 3, resource: { buffer: params } },
    ],
  });
  const run = async (read) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n / 8), Math.ceil(n / 8), 1);
    pass.end();
    if (read) encoder.copyBufferToBuffer(bufferC, 0, readback, 0, 16);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (read) {
      await readback.mapAsync(GPUMapMode.READ);
      const value = new Float32Array(readback.getMappedRange().slice(0))[0];
      readback.unmap();
      if (!Number.isFinite(value)) throw new Error("GPU benchmark returned a non-finite value.");
    }
  };
  await run(false);
  const samples = [];
  for (let i = 0; i < 3; i++) {
    const started = performance.now();
    await run(i === 2);
    samples.push(performance.now() - started);
  }
  samples.sort((x, y) => x - y);
  const medianMs = samples[1];
  const gflops = (2 * n * n * n) / (medianMs / 1000) / 1e9;
  bufferA.destroy(); bufferB.destroy(); bufferC.destroy(); params.destroy(); readback.destroy();
  return { medianMs, gflops, samples };
}

async function runBenchmark() {
  $("run-benchmark").disabled = true;
  $("bench-status").textContent = "Running WebGPU compute tests…";
  const dtype = device.features.has("shader-f16") ? "f16" : "f32";
  $("bench-precision").textContent = dtype.toUpperCase() + " inputs / FP32 accumulation";
  log("GPU matrix benchmark started · " + dtype);
  try {
    for (const n of [256, 512]) {
      const result = await runOneMatmul(n, dtype);
      $( "bench-" + n ).textContent = result.medianMs.toFixed(1) + " ms · " + result.gflops.toFixed(2) + " GFLOP/s";
      log("PERF · " + n + "×" + n + " matmul · median " + result.medianMs.toFixed(1) +
        " ms · " + result.gflops.toFixed(2) + " GFLOP/s · samples " + result.samples.map((v) => v.toFixed(1)).join(", ") + " ms");
    }
    $("bench-status").textContent = "Complete. This measures a simple matrix kernel, not the complete FLUX transformer.";
  } catch (error) {
    $("bench-status").textContent = "Benchmark failed: " + error.message;
    log("ERROR · benchmark · " + (error.stack || error.message));
  } finally {
    $("run-benchmark").disabled = false;
  }
}

$("refresh").addEventListener("click", inspectDevice);
$("run-benchmark").addEventListener("click", runBenchmark);
$("clear-log").addEventListener("click", () => { $("log").textContent = ""; });
$("copy-report").addEventListener("click", async () => {
  const report = {
    ...lastReport,
    benchmark: {
      precision: $("bench-precision").textContent,
      size256: $("bench-256").textContent,
      size512: $("bench-512").textContent,
    },
    log: $("log").textContent,
  };
  const text = JSON.stringify(report, null, 2);
  try {
    await navigator.clipboard.writeText(text);
    $("copy-status").textContent = "Copied.";
  } catch {
    $("copy-status").textContent = "Clipboard unavailable; select and copy the runtime log.";
    log("DIAGNOSTIC JSON\n" + text);
  }
});

inspectDevice();
