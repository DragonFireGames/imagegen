#!/usr/bin/env python3
"""Export NoobAI XL 1.1's SDXL components to a browser-oriented ONNX pack.

OFFLINE CONVERSION ONLY. End users do not need Python or CUDA. The exporter
uses a CUDA machine for tracing, then streams ONNX external weights into <=1.5
GB files so the browser's ONNX Runtime JSPI path can read them from OPFS.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import uuid
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper
from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor
from huggingface_hub import model_info

DEFAULT_MODEL = "Laxhar/noobai-XL-1.1"
SHARD_BYTES = 1_500_000_000
COPY_BUFFER_BYTES = 8 * 1024 * 1024
OPSET = 17


class TextEncoderOne(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        output = self.encoder(input_ids, output_hidden_states=True, return_dict=True)
        return output.hidden_states[-2].to(torch.float32)


class TextEncoderTwo(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        output = self.encoder(input_ids, output_hidden_states=True, return_dict=True)
        return output.hidden_states[-2].to(torch.float32), output.text_embeds.to(torch.float32)


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )[0]


class VAEDecoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent_sample):
        # The JavaScript sampler applies the model's VAE scaling factor.
        return self.vae.decode(latent_sample, return_dict=False)[0].to(torch.float32)


def export_one(wrapper, args, folder: Path, input_names, output_names, dynamic_axes):
    folder.mkdir(parents=True, exist_ok=True)
    graph = folder / "model.onnx"
    wrapper.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            str(graph),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            do_constant_folding=True,
            export_params=True,
            external_data=True,
            dynamo=False,
        )
    return graph


def upcast_fragile_ops(graph_path: Path):
    """Upcast U-Net normalization/softmax math without loading its weights to RAM.

    ONNX initializers remain external-data references while only graph nodes
    are edited. This is important because the NoobAI U-Net is >5 GB.
    """
    model = onnx.load(str(graph_path), load_external_data=False)
    graph = model.graph
    new_nodes = []
    changed = 0

    for node in list(graph.node):
        if node.op_type not in {"LayerNormalization", "InstanceNormalization", "Softmax"}:
            new_nodes.append(node)
            continue

        changed += 1
        for index, input_name in enumerate(list(node.input)):
            if not input_name:
                continue
            temp = f"{input_name}_fp32_{changed}_{index}"
            new_nodes.append(helper.make_node(
                "Cast", [input_name], [temp], to=TensorProto.FLOAT,
                name=f"noobai_fp32_input_{changed}_{index}",
            ))
            node.input[index] = temp

        original_outputs = list(node.output)
        for index, output_name in enumerate(original_outputs):
            if output_name:
                node.output[index] = f"{output_name}_fp32_{changed}_{index}"
        new_nodes.append(node)
        for index, output_name in enumerate(original_outputs):
            if output_name:
                new_nodes.append(helper.make_node(
                    "Cast", [node.output[index]], [output_name],
                    to=TensorProto.FLOAT16,
                    name=f"noobai_fp16_output_{changed}_{index}",
                ))

    del graph.node[:]
    graph.node.extend(new_nodes)
    onnx.save_model(model, str(graph_path), save_as_external_data=False)
    print(f"[onnx] {graph_path.parent.name}: upcast {changed} normalization/softmax ops")


def _external_fields(tensor):
    return {item.key: item.value for item in tensor.external_data}


def reshard_external_weights(graph_path: Path, max_shard_bytes=SHARD_BYTES):
    """Repack external tensor data into <=1.5-GB files using bounded RAM.

    Handles both ONNX external-data tensors and any larger raw_data initializers.
    Each tensor is kept contiguous in a single shard as required by ONNX.
    """
    model = onnx.load(str(graph_path), load_external_data=False)
    graph = model.graph
    folder = graph_path.parent
    old_data_files = list(folder.glob("*.onnx_data*")) + list(folder.glob("*.onnx.data")) + list(folder.glob("*.data"))

    shards = []
    shard_index = -1
    written = 0
    handle = None

    def open_next():
        nonlocal shard_index, written, handle
        if handle:
            handle.flush()
            handle.close()
        shard_index += 1
        written = 0
        name = f"weights-{shard_index:03d}.bin"
        handle = open(folder / name, "wb")
        shards.append(name)

    def begin_tensor(length):
        nonlocal written
        if length > max_shard_bytes:
            raise RuntimeError(
                f"One ONNX initializer is {length} bytes, larger than the {max_shard_bytes}-byte shard limit."
            )
        if handle is None or (written and written + length > max_shard_bytes):
            open_next()
        offset = written
        return offset

    def write_tensor(tensor):
        nonlocal written
        external = _external_fields(tensor)
        raw = tensor.raw_data if tensor.HasField("raw_data") else None

        if external:
            location = external.get("location")
            if not location:
                raise RuntimeError(f"External initializer {tensor.name} has no location.")
            source_path = (folder / location).resolve()
            offset = int(external.get("offset", "0"))
            length = int(external.get("length", str(source_path.stat().st_size - offset)))
            if length < 0 or offset < 0 or offset + length > source_path.stat().st_size:
                raise RuntimeError(f"External-data range is invalid for initializer {tensor.name}.")
            shard_offset = begin_tensor(length)
            with open(source_path, "rb") as source:
                source.seek(offset)
                remaining = length
                while remaining:
                    chunk = source.read(min(COPY_BUFFER_BYTES, remaining))
                    if not chunk:
                        raise RuntimeError(f"Unexpected EOF while copying initializer {tensor.name}.")
                    handle.write(chunk)
                    remaining -= len(chunk)
            written += length
        elif raw is not None and len(raw) > 4096:
            length = len(raw)
            shard_offset = begin_tensor(length)
            handle.write(raw)
            written += length
        else:
            return

        del tensor.external_data[:]
        for key, value in (
            ("location", shards[-1]),
            ("offset", str(shard_offset)),
            ("length", str(length)),
        ):
            entry = tensor.external_data.add()
            entry.key = key
            entry.value = value
        tensor.data_location = TensorProto.EXTERNAL
        tensor.ClearField("raw_data")

    try:
        for tensor in graph.initializer:
            write_tensor(tensor)
        if handle:
            handle.flush()
            handle.close()
            handle = None
        onnx.save_model(model, str(graph_path), save_as_external_data=False)
    except Exception:
        if handle:
            handle.close()
        for name in shards:
            (folder / name).unlink(missing_ok=True)
        raise

    # Remove only the source external-data assets, after the new ONNX graph has
    # been committed with its replacement references.
    for old in old_data_files:
        old.unlink(missing_ok=True)
    return shards


def export_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA runtime is required for this conversion. Use a CUDA machine "
            "with about 16 GB VRAM and 24+ GB host RAM. This is conversion-only; "
            "ordinary users do not need Python or CUDA to run the published browser pack."
        )

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # Resolve the mutable branch/tag to a commit so the ONNX weights and
    # browser-downloaded tokenizer files always refer to the same source state.
    info = model_info(args.model, revision=args.revision)
    source_revision = info.sha
    print(f"[load] {args.model}@{source_revision} as float16")
    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        revision=source_revision,
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    # Trace the explicit attention path; PyTorch's fused SDPA op is not
    # reliably exportable through the legacy ONNX tracing interface.
    pipe.unet.set_attn_processor(AttnProcessor())

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ids = torch.zeros((1, 77), dtype=torch.int64, device="cuda")

    print("[export] CLIP-L")
    te1 = TextEncoderOne(pipe.text_encoder).to("cuda").eval()
    export_one(
        te1, (ids,), output / "text_encoder",
        ["input_ids"], ["hidden_states"],
        {"input_ids": {0: "batch"}, "hidden_states": {0: "batch"}},
    )
    del te1
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()

    print("[export] CLIP-bigG")
    te2 = TextEncoderTwo(pipe.text_encoder_2).to("cuda").eval()
    export_one(
        te2, (ids,), output / "text_encoder_2",
        ["input_ids"], ["hidden_states", "text_embeds"],
        {
            "input_ids": {0: "batch"},
            "hidden_states": {0: "batch"},
            "text_embeds": {0: "batch"},
        },
    )
    del te2
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()

    print("[export] SDXL U-Net (1024px reference shape; dynamic spatial axes)")
    unet = UNetWrapper(pipe.unet).to("cuda").eval()
    latent = torch.randn((1, 4, 128, 128), dtype=torch.float16, device="cuda")
    timestep = torch.tensor([999.0], dtype=torch.float16, device="cuda")
    hidden = torch.randn((1, 77, 2048), dtype=torch.float16, device="cuda")
    pooled = torch.randn((1, 1280), dtype=torch.float16, device="cuda")
    time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.float16, device="cuda")
    unet_graph = export_one(
        unet, (latent, timestep, hidden, pooled, time_ids), output / "unet",
        ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
        ["noise_pred"],
        {
            "sample": {0: "batch", 2: "latent_height", 3: "latent_width"},
            "encoder_hidden_states": {0: "batch"},
            "text_embeds": {0: "batch"},
            "time_ids": {0: "batch"},
            "noise_pred": {0: "batch", 2: "latent_height", 3: "latent_width"},
        },
    )
    del unet, latent, timestep, hidden, pooled, time_ids
    del pipe.unet
    torch.cuda.empty_cache()
    gc.collect()
    upcast_fragile_ops(unet_graph)

    print("[export] VAE decoder in fp32")
    scaling_factor = float(pipe.vae.config.scaling_factor)
    vae = VAEDecoder(pipe.vae).to(device="cuda", dtype=torch.float32).eval()
    vae_input = torch.randn((1, 4, 128, 128), dtype=torch.float32, device="cuda")
    vae_graph = export_one(
        vae, (vae_input,), output / "vae_decoder",
        ["latent_sample"], ["sample"],
        {
            "latent_sample": {0: "batch", 2: "latent_height", 3: "latent_width"},
            "sample": {0: "batch", 2: "image_height", 3: "image_width"},
        },
    )
    del vae, vae_input
    torch.cuda.empty_cache()
    gc.collect()

    scheduler = dict(pipe.scheduler.config)
    del pipe
    gc.collect()

    print("[package] streaming ONNX external weights into <= 1.5 GB shards")
    component_data = {}
    for component in ("text_encoder", "text_encoder_2", "unet", "vae_decoder"):
        graph = output / component / "model.onnx"
        shards = reshard_external_weights(graph)
        component_data[component] = {
            "graph": f"{component}/model.onnx",
            "externalData": [
                {
                    "path": name,
                    "asset": f"{component}/{name}",
                    "bytes": (output / component / name).stat().st_size,
                }
                for name in shards
            ],
            "graphBytes": graph.stat().st_size,
            "graphBytesAsset": f"{component}/model.onnx",
        }

    manifest = {
        "format": "noobai-sdxl-browser-onnx-v1",
        "packId": f"{source_revision[:12]}-{uuid.uuid4().hex[:8]}",
        "model": "NoobAI XL 1.1",
        "sourceModel": args.model,
        "sourceRevision": source_revision,
        "pipeline": "StableDiffusionXLPipeline",
        "opset": OPSET,
        "weightDtype": "float16",
        "sequenceLength": 77,
        "latentChannels": 4,
        "vaeScalingFactor": scaling_factor,
        "scheduler": scheduler,
        "components": component_data,
        "runtime": {
            "name": "onnxruntime-web",
            "version": "1.31.1",
            "entry": "jspi",
            "textEncoderExecutionProvider": "wasm",
            "unetExecutionProvider": "webgpu",
            "vaeExecutionProvider": "webgpu",
        },
        "notes": [
            "Text encoders expose penultimate hidden states; encoder 2 also exposes pooled text_embeds.",
            "U-Net inputs and outputs use float16. Text encoder outputs and VAE input/output use float32.",
            "The VAE scaling factor is applied in the JavaScript sampler before decoding.",
            "Tokenizer assets are fetched at the pinned sourceRevision from sourceModel.",
            "Only the VAE decoder is included; this pack targets text-to-image generation.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total_size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"[done] browser pack created at {output}")
    print(f"[done] total size: {total_size / 1e9:.2f} GB")

    if args.upload_repo:
        from huggingface_hub import HfApi
        token = args.token or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("--upload-repo requires --token or HF_TOKEN.")
        api = HfApi(token=token)
        api.create_repo(args.upload_repo, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.upload_repo,
            repo_type="model",
            folder_path=str(output),
            path_in_repo=args.path_in_repo,
            commit_message="Upload NoobAI XL 1.1 browser ONNX pack",
        )
        print(f"[upload] https://huggingface.co/{args.upload_repo}/tree/main/{args.path_in_repo}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="NoobAI XL 1.1 Diffusers model repo.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", default="noobai-xl-1.1-browser-pack")
    parser.add_argument("--upload-repo", help="Optional Hugging Face model repo to publish the browser ONNX pack.")
    parser.add_argument("--path-in-repo", default="browser-pack")
    parser.add_argument("--token", help="Hugging Face write token; alternatively set HF_TOKEN.")
    parser.add_argument("--private", action="store_true", help="Create target HF repo as private.")
    return parser.parse_args()


if __name__ == "__main__":
    export_model(parse_args())
