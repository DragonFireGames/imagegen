#!/usr/bin/env python3
"""Fuse an SDXL LoRA into Illustrious XL v0.1 and optionally export ONNX.

The Python environment is an offline build/reference tool. It is not used by
the browser at inference time. Model weights and generated exports are never
written into this Git repository by default.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter


DEFAULT_BASE = "OnomaAIResearch/Illustrious-xl-early-release-v0"
ADAPTER_NAME = "personal-ami"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse a local SDXL LoRA into Illustrious and optionally export ONNX."
    )
    parser.add_argument("--lora", required=True, type=Path, help="Path to the local .safetensors LoRA file.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory (keep this outside the Git repo).")
    parser.add_argument("--base-model", default=DEFAULT_BASE, help=f"Diffusers base model (default: {DEFAULT_BASE}).")
    parser.add_argument("--strength", type=float, default=0.75, help="LoRA fuse scale (suggested Personal Ami range: 0.6–0.9).")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16", help="Illustrious v0.1 is published with BF16 weights; use float16 for hardware without reliable BF16 support.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto", help="Device used for fusion/export. Auto prefers CUDA, then Apple MPS, then CPU.")
    parser.add_argument("--export-onnx", action="store_true", help="Run optimum-cli export after fusing.")
    parser.add_argument("--keep-diffusers", action="store_true", help="Keep the merged Diffusers model beside the ONNX export.")
    parser.add_argument("--adapter-name", default=ADAPTER_NAME, help="Internal Diffusers adapter name.")
    return parser.parse_args()



def inspect_lora_header(lora_path: Path, safe_open) -> dict:
    """Inspect tensor keys/shapes and embedded metadata without loading payloads."""
    component_prefixes = {
        "lora_unet_": "unet",
        "lora_te1_": "text_encoder",
        "lora_te2_": "text_encoder_2",
    }
    suffixes = (
        (".lora_down.weight", "down"),
        (".lora_up.weight", "up"),
        (".lora_A.weight", "down"),
        (".lora_B.weight", "up"),
    )
    component_counts = Counter()
    pair_shapes = {}
    alpha_count = 0
    unknown_keys = []
    tensor_count = 0

    with safe_open(str(lora_path), framework="pt", device="cpu") as handle:
        header_metadata = handle.metadata() or {}
        keys = list(handle.keys())
        tensor_count = len(keys)

        for key in keys:
            component = next(
                (name for prefix, name in component_prefixes.items() if key.startswith(prefix)),
                None,
            )
            if component is None:
                unknown_keys.append(key)
            else:
                component_counts[component] += 1

            if key.endswith(".alpha"):
                alpha_count += 1
                continue

            pair_suffix = next(
                ((suffix, side) for suffix, side in suffixes if key.endswith(suffix)),
                None,
            )
            if pair_suffix is None:
                continue

            suffix, side = pair_suffix
            stem = key[:-len(suffix)]
            shape = tuple(handle.get_slice(key).get_shape())
            pair_shapes.setdefault(stem, {})[side] = shape

    incomplete = 0
    rank_mismatches = []
    ranks = Counter()
    for stem, shapes in pair_shapes.items():
        down = shapes.get("down")
        up = shapes.get("up")
        if down is None or up is None:
            incomplete += 1
            continue
        if len(down) < 2 or len(up) < 2 or down[0] != up[1]:
            rank_mismatches.append({"layer": stem, "down": down, "up": up})
            continue
        ranks[down[0]] += 1

    expected_components = {"unet", "text_encoder", "text_encoder_2"}
    missing_components = sorted(expected_components - set(component_counts))
    if missing_components:
        raise RuntimeError(
            "The LoRA is missing expected SDXL component groups: "
            + ", ".join(missing_components)
        )
    if unknown_keys:
        sample = ", ".join(unknown_keys[:8])
        raise RuntimeError(
            f"The LoRA contains {len(unknown_keys)} keys outside the supported "
            f"SDXL Kohya prefixes (examples: {sample})."
        )
    if incomplete or rank_mismatches:
        raise RuntimeError(
            f"LoRA header validation failed: {incomplete} incomplete pairs, "
            f"{len(rank_mismatches)} rank mismatches."
        )

    return {
        "metadata": header_metadata,
        "tensor_count": tensor_count,
        "component_tensor_counts": dict(component_counts),
        "pair_count": len(pair_shapes),
        "alpha_tensor_count": alpha_count,
        "rank_counts": {str(rank): count for rank, count in ranks.items()},
    }


def choose_device(torch, requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda was selected, but CUDA is not available.")
        if requested == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise SystemExit("--device mps was selected, but Apple MPS is not available.")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    args = parse_args()
    lora_path = args.lora.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not lora_path.is_file():
        raise SystemExit(f"LoRA file not found: {lora_path}")
    if lora_path.suffix.lower() != ".safetensors":
        raise SystemExit("Only .safetensors LoRA files are accepted by this script.")
    if not 0.0 <= args.strength <= 2.0:
        raise SystemExit("--strength must be between 0.0 and 2.0.")
    if output_dir.exists() and not output_dir.is_dir():
        raise SystemExit(f"Output path exists and is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory must be empty: {output_dir}")
    if output_dir == Path.cwd().resolve() or Path.cwd().resolve() in output_dir.parents:
        # Keeping the output outside this repo is intentional: exports are multi-GB
        # model artifacts and must not accidentally be committed.
        print("WARNING: output is inside the current working tree. Keep model weights out of Git.", file=sys.stderr)

    try:
        import torch
        import diffusers
        from diffusers import DiffusionPipeline
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as exc:
        raise SystemExit(
            "Missing export dependencies. Install tools/requirements-export.txt "
            "in a dedicated Python environment after installing a matching PyTorch build."
        ) from exc

    device = choose_device(torch, args.device)
    lora_header = inspect_lora_header(lora_path, safe_open)
    print("LoRA header:", flush=True)
    print(json.dumps({
        "tensor_count": lora_header["tensor_count"],
        "component_tensor_counts": lora_header["component_tensor_counts"],
        "pair_count": lora_header["pair_count"],
        "alpha_tensor_count": lora_header["alpha_tensor_count"],
        "rank_counts": lora_header["rank_counts"],
        "ss_base_model_version": lora_header["metadata"].get("ss_base_model_version"),
        "ss_network_dim": lora_header["metadata"].get("ss_network_dim"),
        "ss_network_alpha": lora_header["metadata"].get("ss_network_alpha"),
    }, indent=2), flush=True)
    if lora_header["pair_count"] != lora_header["alpha_tensor_count"]:
        print("WARNING: LoRA pair count and .alpha tensor count differ.", file=sys.stderr)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = output_dir.parent
    with tempfile.TemporaryDirectory(prefix="illustrious-lora-fuse-", dir=staging_parent) as temp_name:
        merged_dir = Path(temp_name) / "merged-diffusers"

        print(f"Loading base pipeline: {args.base_model}", flush=True)
        print(f"Loading dtype: {args.dtype}", flush=True)
        print(f"Fusion device: {device}", flush=True)
        pipe = DiffusionPipeline.from_pretrained(
            args.base_model,
            dtype=dtype_map[args.dtype],
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        pipe.to(device)

        print(f"Reading local adapter tensors: {lora_path.name}", flush=True)
        # Diffusers accepts a state dict directly. Passing the .safetensors file
        # itself as the model path is ambiguous and is not the documented local
        # file interface. This also preserves Kohya .alpha entries for conversion
        # into Diffusers network_alphas.
        lora_state_dict = load_file(str(lora_path), device="cpu")
        pipe.load_lora_weights(
            lora_state_dict,
            adapter_name=args.adapter_name,
            low_cpu_mem_usage=True,
        )
        del lora_state_dict

        adapter_map = pipe.get_list_adapters()
        attached_components = [
            component
            for component, names in adapter_map.items()
            if args.adapter_name in names
        ]
        required_components = {"unet", "text_encoder", "text_encoder_2"}
        missing_components = sorted(required_components - set(attached_components))
        if missing_components:
            raise RuntimeError(
                "Diffusers did not attach all three components reported by the "
                "header. Missing: " + ", ".join(missing_components)
            )

        print(
            "Adapter found in: " + ", ".join(attached_components),
            flush=True,
        )
        print(f"Fusing at strength {args.strength:g} with safe_fusing=True...", flush=True)
        pipe.fuse_lora(
            components=["unet", "text_encoder", "text_encoder_2"],
            lora_scale=args.strength,
            safe_fusing=True,
            adapter_names=[args.adapter_name],
        )

        # Fused deltas remain in the base layer weights. Removing the PEFT
        # wrappers makes the saved pipeline a plain Diffusers model again.
        pipe.unload_lora_weights()
        print(f"Saving fused Diffusers pipeline to temporary staging: {merged_dir}", flush=True)
        pipe.save_pretrained(
            merged_dir,
            safe_serialization=True,
            max_shard_size="2GB",
        )
        del pipe

        metadata = {
            "base_model": args.base_model,
            "lora_file": lora_path.name,
            "adapter_name": args.adapter_name,
            "lora_strength": args.strength,
            "source_dtype": args.dtype,
            "device": device,
            "fused_components": attached_components,
            "lora_header_metadata": lora_header["metadata"],
            "lora_tensor_count": lora_header["tensor_count"],
            "lora_component_tensor_counts": lora_header["component_tensor_counts"],
            "lora_pair_count": lora_header["pair_count"],
            "lora_alpha_tensor_count": lora_header["alpha_tensor_count"],
            "lora_rank_counts": lora_header["rank_counts"],
            "diffusers_version": diffusers.__version__,
            "torch_version": torch.__version__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "note": "Fused reference model; browser WebGPU compatibility is not implied by this export.",
        }

        if args.export_onnx:
            cli = shutil.which("optimum-cli")
            if not cli:
                raise SystemExit(
                    "optimum-cli was not found. Install tools/requirements-export.txt "
                    "or rerun without --export-onnx to save the fused Diffusers model."
                )
            print("Exporting SDXL ONNX components with Optimum (fp16)...", flush=True)
            command = [
                cli,
                "export",
                "onnx",
                "--model",
                str(merged_dir),
                "--task",
                "stable-diffusion-xl",
                "--dtype",
                "fp16",
                str(output_dir),
            ]
            print("Running: " + " ".join(command), flush=True)
            subprocess.run(command, check=True)
            metadata["output_format"] = "optimum-sdxl-onnx-fp16"
            metadata["browser_ready"] = False
            if args.keep_diffusers:
                keep_path = output_dir / "merged-diffusers"
                shutil.copytree(merged_dir, keep_path)
                print(f"Kept merged Diffusers model: {keep_path}", flush=True)
        else:
            # When ONNX export is not requested, the requested output is the
            # fused Diffusers directory itself. Move staging contents atomically.
            shutil.copytree(merged_dir, output_dir, dirs_exist_ok=True)
            metadata["output_format"] = "diffusers"
            metadata["browser_ready"] = False

    metadata_path = output_dir / "export-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Export metadata: {metadata_path}", flush=True)
    print("Finished. Compare outputs against the same base + LoRA in Diffusers before browser integration.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
