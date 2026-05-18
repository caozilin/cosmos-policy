#!/usr/bin/env python3
"""
Extract all unique prompts from VLA4Desk collected data.json files.

Usage:
    # 查看所有提取到的 prompt
    python scripts/extract_vla4desk_prompts.py

    # 只列 prompt（不输出 task 信息）
    python scripts/extract_vla4desk_prompts.py --quiet
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cosmos_policy.config.output_paths import default_vla4desk_collected_dir  # noqa: E402


def collect_prompts(data_root: str) -> OrderedDict:
    """Walk through all data.json and return OrderedDict {prompt: task_name}."""
    prompts = OrderedDict()
    for root, dirs, files in os.walk(data_root):
        for f in files:
            if f == "data.json":
                fp = os.path.join(root, f)
                with open(fp) as fh:
                    data = json.load(fh)
                p = data.get("prompt", "").strip()
                if p and p not in prompts:
                    prompts[p] = data.get("task_name", "")
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Extract VLA4Desk prompts")
    parser.add_argument(
        "--data-root",
        type=str,
        default=default_vla4desk_collected_dir(),
        help="Root of collected episodes (default: <repo>/vla4desk/collected)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Only list prompts, no task info")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save prompts to a text file (one per line)")
    args = parser.parse_args()

    prompts = collect_prompts(args.data_root)

    if args.output:
        with open(args.output, "w") as f:
            for p in prompts:
                f.write(p + "\n")
        print(f"Saved {len(prompts)} prompts to {args.output}")
        return

    print(f"Total unique prompts: {len(prompts)}")
    print(f"Total tasks: {len(set(prompts.values()))}")
    print()

    if args.quiet:
        for p in prompts:
            print(p)
    else:
        for i, (p, task) in enumerate(prompts.items(), 1):
            print(f"{i:3d}. [{task}] {p}")


if __name__ == "__main__":
    main()
