#!/usr/bin/env python3
"""
Extract unique prompts from eval telemetry.jsonl files.

Can merge with prior prompt lists (e.g. original 135 collected prompts + eval).

Usage:
    # eval only
    python scripts/extract_eval_prompts.py

    # 原 135 条 (git 里 vla4desk/prompts.txt) + eval，合并去重
    python scripts/extract_eval_prompts.py --merge-committed-prompts

    # 指定额外文件先合并，再扫 eval
    python scripts/extract_eval_prompts.py --merge-from path/to/old_prompts.txt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import OrderedDict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_EVAL_ROOT = "/home/czl/桌面/毕设/结题报告/素材/eval"
_DEFAULT_OUTPUT = os.path.join(_PROJECT_ROOT, "vla4desk", "prompts.txt")
_COMMITTED_PROMPTS_GIT_PATH = "vla4desk/prompts.txt"


def load_prompts_from_text(path: str) -> OrderedDict[str, None]:
    """Load prompts from a text file (one per line), preserving order."""
    prompts: OrderedDict[str, None] = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p and p not in prompts:
                prompts[p] = None
    return prompts


def load_committed_prompts(repo_root: str) -> OrderedDict[str, None]:
    """Load vla4desk/prompts.txt from git HEAD (the original 135 collected prompts)."""
    try:
        text = subprocess.check_output(
            ["git", "show", f"HEAD:{_COMMITTED_PROMPTS_GIT_PATH}"],
            cwd=repo_root,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Could not read HEAD:{_COMMITTED_PROMPTS_GIT_PATH} via git. "
            f"Use --merge-from with a saved copy of the old prompts.txt instead."
        ) from e
    prompts: OrderedDict[str, None] = OrderedDict()
    for line in text.splitlines():
        p = line.strip()
        if p and p not in prompts:
            prompts[p] = None
    return prompts


def merge_prompts(into: OrderedDict[str, None], extra: OrderedDict[str, None]) -> int:
    """Append keys from extra into into. Returns count of newly added prompts."""
    added = 0
    for p in extra:
        if p not in into:
            into[p] = None
            added += 1
    return added


def collect_prompts_from_telemetry(eval_root: str) -> tuple[OrderedDict[str, None], int, int]:
    """
    Returns (ordered unique prompts, num_telemetry_files, num_jsonl_lines_read).
    """
    if not os.path.isdir(eval_root):
        raise FileNotFoundError(f"Eval root not found: {eval_root}")

    prompts: OrderedDict[str, None] = OrderedDict()
    n_files = 0
    n_lines = 0

    for root, _dirs, files in os.walk(eval_root):
        if "telemetry.jsonl" not in files:
            continue
        fp = os.path.join(root, "telemetry.jsonl")
        n_files += 1
        with open(fp, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {fp}:{line_no}: {e}") from e
                prompt = record.get("prompt", "")
                if isinstance(prompt, str):
                    prompt = prompt.strip()
                else:
                    prompt = str(prompt).strip() if prompt is not None else ""
                if prompt and prompt not in prompts:
                    prompts[prompt] = None

    return prompts, n_files, n_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract unique prompts from eval telemetry.jsonl (optional merge)"
    )
    parser.add_argument(
        "--eval-root",
        type=str,
        default=_DEFAULT_EVAL_ROOT,
        help=f"Root directory to search (default: {_DEFAULT_EVAL_ROOT})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=_DEFAULT_OUTPUT,
        help=f"Output text file, one prompt per line (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--merge-from",
        action="append",
        default=[],
        metavar="FILE",
        help="Merge prompts from FILE first (can repeat); then append eval prompts",
    )
    parser.add_argument(
        "--merge-committed-prompts",
        action="store_true",
        help=f"Merge git HEAD:{_COMMITTED_PROMPTS_GIT_PATH} (original 135) before eval",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only print summary, not each prompt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print merged prompts to stdout without writing the output file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = os.path.abspath(os.path.expanduser(args.eval_root))
    output_path = os.path.abspath(os.path.expanduser(args.output))

    merged: OrderedDict[str, None] = OrderedDict()
    n_base = 0

    if args.merge_committed_prompts:
        base = load_committed_prompts(_PROJECT_ROOT)
        n_base += merge_prompts(merged, base)
        if not args.quiet:
            print(f"Merged git HEAD:{_COMMITTED_PROMPTS_GIT_PATH}: {len(base)} prompts ({n_base} new)")

    for path in args.merge_from:
        path = os.path.abspath(os.path.expanduser(path))
        base = load_prompts_from_text(path)
        added = merge_prompts(merged, base)
        n_base += added
        if not args.quiet:
            print(f"Merged {path}: {len(base)} lines, +{added} new")

    eval_prompts, n_files, n_lines = collect_prompts_from_telemetry(eval_root)
    n_eval_new = merge_prompts(merged, eval_prompts)

    if not args.quiet:
        print(f"Eval root: {eval_root}")
        print(f"telemetry.jsonl files: {n_files}")
        print(f"JSONL lines read: {n_lines}")
        print(f"Eval unique prompts: {len(eval_prompts)} (+{n_eval_new} new vs merged base)")
        print(f"Total after merge: {len(merged)}")
        print()

    if not merged:
        print("No prompts found.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        for p in merged:
            print(p)
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in merged:
            f.write(p + "\n")

    print(f"Saved {len(merged)} unique prompts to {output_path}")
    if not args.quiet:
        for i, p in enumerate(merged, 1):
            print(f"{i:3d}. {p}")


if __name__ == "__main__":
    main()
