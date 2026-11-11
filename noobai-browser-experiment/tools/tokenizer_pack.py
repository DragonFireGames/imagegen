#!/usr/bin/env python3
"""Create and publish Transformers.js-compatible fast tokenizers for a NoobAI pack."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer


def save_fast_tokenizers(model_id: str, revision: str, output: Path) -> None:
    """Convert both SDXL CLIP tokenizers to fast-tokenizer JSON files."""
    for folder in ("tokenizer", "tokenizer_2"):
        print(f"[tokenizer] Converting {model_id}/{folder}@{revision}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            subfolder=folder,
            revision=revision,
            use_fast=True,
        )
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError(f"{folder} did not load as a fast tokenizer.")
        target = output / folder
        target.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(target)
        tokenizer_json = target / "tokenizer.json"
        if not tokenizer_json.is_file() or tokenizer_json.stat().st_size == 0:
            raise RuntimeError(f"{folder} did not produce tokenizer.json.")
        print(f"[tokenizer] Wrote {tokenizer_json} ({tokenizer_json.stat().st_size:,} bytes)")


def repair_pack(args) -> None:
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("A Hugging Face write token is required (--token or HF_TOKEN).")

    manifest_path = f"{args.path_in_repo}/manifest.json".strip("/")
    manifest_file = hf_hub_download(
        repo_id=args.upload_repo,
        repo_type="model",
        filename=manifest_path,
        token=token,
    )
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    source_model = manifest["sourceModel"]
    source_revision = manifest["sourceRevision"]

    with tempfile.TemporaryDirectory(prefix="noobai-tokenizers-") as temporary:
        output = Path(temporary)
        save_fast_tokenizers(source_model, source_revision, output)

        manifest["tokenizerModel"] = args.upload_repo
        manifest["tokenizerRevision"] = "main"
        manifest["tokenizerSubfolders"] = [
            f"{args.path_in_repo}/tokenizer".strip("/"),
            f"{args.path_in_repo}/tokenizer_2".strip("/"),
        ]
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        print("[upload] Adding fast tokenizers and updating manifest; ONNX weights are untouched.")
        HfApi(token=token).upload_folder(
            repo_id=args.upload_repo,
            repo_type="model",
            folder_path=str(output),
            path_in_repo=args.path_in_repo,
            commit_message="Add browser-compatible NoobAI fast tokenizers",
        )

    print(f"[done] Updated {args.upload_repo}/{manifest_path}")
    print("[done] No U-Net, text encoder, or VAE ONNX conversion was rerun.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload-repo", required=True, help="Existing public Hugging Face model repo containing the browser pack.")
    parser.add_argument("--path-in-repo", default="browser-pack")
    parser.add_argument("--token", help="Hugging Face write token; alternatively set HF_TOKEN.")
    repair_pack(parser.parse_args())


if __name__ == "__main__":
    main()
