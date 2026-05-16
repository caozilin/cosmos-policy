# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Precomputes T5 text embeddings for VLA4Desk task descriptions from collected data.json files.

Usage:
    uv run -m cosmos_policy.datasets.save_vla4desk_t5_text_embeddings
"""

import argparse
import json
import os
from collections import OrderedDict

from cosmos_policy.datasets.t5_embedding_utils import (
    generate_t5_embeddings,
    save_embeddings,
)


def collect_prompts(data_root: str) -> list[str]:
    """Walk through all data.json files and collect unique prompts."""
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute T5 text embeddings for VLA4Desk task descriptions"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/media/czl/sata/franka_my_code/vla4desk/collected",
        help="Root directory containing task subdirectories with data.json files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="users/user/data/vla4desk",
        help="Directory to save the T5 embeddings pkl file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Collect unique prompts
    print("Collecting prompts from data.json files...")
    prompts = collect_prompts(args.data_root)
    unique_commands = list(prompts.keys())
    print(f"Found {len(unique_commands)} unique prompts across {len(set(prompts.values()))} tasks:")
    for task_name in sorted(set(prompts.values())):
        count = sum(1 for v in prompts.values() if v == task_name)
        print(f"  {task_name}: {count} prompts")

    # Generate T5 embeddings
    t5_text_embeddings = generate_t5_embeddings(unique_commands)
    save_embeddings(t5_text_embeddings, args.output_dir)


if __name__ == "__main__":
    main()
