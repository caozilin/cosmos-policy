#!/usr/bin/env python3
"""
One-time export: T5-11B encoder-only weights for text embedding precompute.

Reads the full ``google-t5/t5-11b`` checkpoint (~45G fp32 on disk), writes a
smaller directory (``hf_cache/t5-11b-encoder/``, ~22G fp32 or ~11G bf16) that
``get_t5_emb.py`` will prefer automatically. Lowers peak CPU RAM on later loads.

Usage:
    export HF_HUB_CACHE=./hf_cache HF_HUB_OFFLINE=1
    python scripts/export_t5_encoder_only.py
    python scripts/export_t5_encoder_only.py --dtype bf16
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cosmos_policy._src.predict2.inference.get_t5_emb import (  # noqa: E402
    T5_ENCODER_SUBDIR,
    T5_HF_REPO,
    assert_t5_hub_cache,
    resolve_hf_hub_cache,
    resolve_torch_dtype,
)
from transformers import T5EncoderModel, T5TokenizerFast  # noqa: E402


def _full_t5_snapshot_dir(cache_dir: str) -> str:
    for path in sorted(glob.glob(os.path.join(cache_dir, "models--google-t5--t5-11b", "snapshots", "*"))):
        if os.path.isfile(os.path.join(path, "pytorch_model.bin")) or os.path.islink(
            os.path.join(path, "pytorch_model.bin")
        ):
            return path
    raise FileNotFoundError(f"No complete T5 snapshot under {cache_dir}/models--google-t5--t5-11b/snapshots/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export T5-11B encoder-only checkpoint")
    p.add_argument(
        "--hf-hub-cache",
        type=str,
        default=None,
        help="HF hub cache root (default: HF_HUB_CACHE or ./hf_cache)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output dir (default: <hf_cache>/{T5_ENCODER_SUBDIR})",
    )
    p.add_argument(
        "--dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Saved weight dtype (default: fp32, matches full checkpoint)",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace existing export")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = resolve_hf_hub_cache(args.hf_hub_cache)
    if cache_dir is None:
        raise FileNotFoundError("Set HF_HUB_CACHE or place weights under ./hf_cache")

    out_dir = args.output_dir or os.path.join(cache_dir, T5_ENCODER_SUBDIR)
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    torch_dtype = resolve_torch_dtype(args.dtype)

    weights_path = os.path.join(out_dir, "pytorch_model.bin")
    if os.path.exists(out_dir) and not args.overwrite:
        if os.path.isfile(weights_path):
            print(f"Encoder-only export already exists: {out_dir}")
            print("Use --overwrite to re-export.")
            return

    assert_t5_hub_cache(cache_dir, model_name=T5_HF_REPO)
    src_snap = _full_t5_snapshot_dir(cache_dir)

    print(f"Loading encoder from full checkpoint (dtype={torch_dtype})...")
    print("Peak CPU RAM may be high during this one-time export.")
    encoder = T5EncoderModel.from_pretrained(
        T5_HF_REPO,
        cache_dir=cache_dir,
        local_files_only=True,
        torch_dtype=torch_dtype,
        use_safetensors=False,
        low_cpu_mem_usage=True,
    )
    os.makedirs(out_dir, exist_ok=True)
    encoder.save_pretrained(out_dir, safe_serialization=False)

    for name in ("spiece.model", "tokenizer.json"):
        shutil.copy2(os.path.join(src_snap, name), os.path.join(out_dir, name))

    T5TokenizerFast.from_pretrained(out_dir, local_files_only=True)

    size_gib = os.path.getsize(weights_path) / (1024**3)
    print(f"Saved encoder-only model to {out_dir}")
    print(f"  pytorch_model.bin: {size_gib:.2f} GiB ({args.dtype})")
    print("Precompute will use this path automatically when present.")


if __name__ == "__main__":
    main()
