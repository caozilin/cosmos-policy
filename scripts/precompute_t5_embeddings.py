#!/usr/bin/env python3
"""
Precompute T5-11B text embeddings from a prompts file (one prompt per line).

Requires project env from ``uv sync --extra cu128 --python 3.10`` (see FRANKA.md).

Usage (from project root, after ``source .venv/bin/activate``):
    HF_HUB_CACHE=./hf_cache HF_HUB_OFFLINE=1 \\
      python scripts/precompute_t5_embeddings.py --gpu 0 --device auto --max_gpu_mem_gib 12
    HF_HUB_CACHE=./hf_cache HF_HUB_OFFLINE=1 \\
      python scripts/precompute_t5_embeddings.py --gpu '' --device cpu
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_INPUT = os.path.join(_PROJECT_ROOT, "vla4desk", "prompts.txt")
_DEFAULT_OUTPUT = os.path.join(_PROJECT_ROOT, "vla4desk", "t5_embeddings.pkl")


def _apply_gpu_env_from_argv() -> None:
    """Set CUDA_VISIBLE_DEVICES before torch is imported."""
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            return
        if arg.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.split("=", 1)[1]
            return


_apply_gpu_env_from_argv()

from cosmos_policy.datasets.t5_embedding_utils import generate_t5_embeddings  # noqa: E402


def load_prompts(path: str) -> list[str]:
    """Load unique non-empty prompts from a text file (one per line)."""
    prompts: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p and p not in seen:
                seen.add(p)
                prompts.append(p)
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute T5 embeddings from a prompts file")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=_DEFAULT_INPUT,
        help="Prompts text file, one prompt per line (default: vla4desk/prompts.txt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=_DEFAULT_OUTPUT,
        help="Output pickle path (default: vla4desk/t5_embeddings.pkl)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="CUDA device id(s), e.g. 0. Use '' to disable GPU.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="cuda: full GPU; cpu: no GPU; auto: GPU+CPU offload",
    )
    parser.add_argument(
        "--max_gpu_mem_gib",
        type=float,
        default=12.0,
        help="When --device auto, max GPU memory in GiB",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        if args.gpu == "" and args.device != "cpu":
            print("Note: --gpu '' — using --device cpu")
            args.device = "cpu"

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Prompts file not found: {args.input}")

    prompts = load_prompts(args.input)
    print(f"Loaded {len(prompts)} unique prompts from {args.input}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(all visible)')}")
    print(
        f"device={args.device}"
        + (f", max_gpu_mem_gib={args.max_gpu_mem_gib}" if args.device == "auto" else "")
    )

    t5_text_embeddings = generate_t5_embeddings(
        prompts, device=args.device, max_gpu_mem_gib=args.max_gpu_mem_gib
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(t5_text_embeddings, f)
    print(f"Saved {len(t5_text_embeddings)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
