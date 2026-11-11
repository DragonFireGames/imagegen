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
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--export-onnx", action="store_true", help="Run optimum-cli export after fusing.")
    parser.add_argument("--keep-diffusers", action="store_true", help="Keep the merged Diffusers model beside the ONNX export.")
    parser.add_argument("--adapter-name", default=ADAPTER_NAME, help="Internal Diffusers adapter name.")
    return parser.parse_args()


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
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory must be empty: {output_dir}")
    if output_dir == Path.cwd().resolve() or Path.cwd().resolve() in output_dir.parents:
        # Keeping the output outside this repo is intentional: exports are multi-GB
        # model artifacts and must not accidentally be committed.
        print("WARNING: output is inside the current working tree. Keep model weights out of Git.", file=sys.stderr)

    try:
        import torch
        import diffusers
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing export dependencies. Install tools/requirements-export.txt "
            "in a dedicated Python environment after installing a matching PyTorch build."
        ) from exc

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
        pipe = DiffusionPipeline.from_pretrained(
            args.base_model,
            torch_dtype=dtype_map[args.dtype],
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )

        print(f"Loading adapter: {lora_path.name}", flush=True)
        pipe.load_lora_weights(str(lora_path), adapter_name=args.adapter_name)

        adapter_map = pipe.get_list_adapters()
        attached_components = [
            component
            for component, names in adapter_map.items()
            if args.adapter_name in names
        ]
        if not attached_components:
            raise RuntimeError(
                "Diffusers loaded no adapter layers for this file. "
                "The file may not be a compatible SDXL LoRA."
            )

        print(
            "Adapter found in: " + ", ".join(attached_components),
            flush=True,
        )
        print(f"Fusing at strength {args.strength:g} with safe_fusing=True...", flush=True)
        pipe.fuse_lora(
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
            "fused_components": attached_components,
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
