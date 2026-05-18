# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository-local default paths for Cosmos Policy (datasets, checkpoints, HF cache)."""

import os

_APPLIED = False


def repo_root() -> str:
    """Repository root (parent of the `cosmos_policy` package)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_hf_home() -> str:
    """`<repo>/hf_cache/`."""
    return os.path.join(repo_root(), "hf_cache")


def default_base_datasets_dir() -> str:
    """`<repo>/` — datasets live under `datasets/...` relative to this."""
    return repo_root()


def default_checkpoints_root() -> str:
    """`<repo>/checkpoints/` — training job outputs and DCP weights."""
    return os.path.join(repo_root(), "checkpoints")


def default_vla4desk_collected_dir() -> str:
    """Raw VLA4Desk recordings: `<repo>/vla4desk/collected/`."""
    return os.path.join(repo_root(), "vla4desk", "collected")


def default_vla4desk_prompts_txt() -> str:
    """`<repo>/vla4desk/prompts.txt` (tracked in git)."""
    return os.path.join(repo_root(), "vla4desk", "prompts.txt")


def default_vla4desk_eval_dir() -> str:
    """Optional eval telemetry root: `<repo>/vla4desk/eval/` (not in git)."""
    return os.path.join(repo_root(), "vla4desk", "eval")


def default_vla4desk_success_only_dir() -> str:
    """Parent of converted suite: `<repo>/datasets/VLA4Desk-Franka/success_only/`."""
    return os.path.join(repo_root(), "datasets", "VLA4Desk-Franka", "success_only")


def default_vla4desk_franka_dataset_dir() -> str:
    """Converted HDF5 suite: `<repo>/datasets/.../vla4desk_franka/`."""
    return os.path.join(default_vla4desk_success_only_dir(), "vla4desk_franka")


def default_vla4desk_example_data_json() -> str:
    """Example sidecar for validation: `.../simple_pick_place/epo_1/data.json`."""
    return os.path.join(default_vla4desk_collected_dir(), "simple_pick_place", "epo_1", "data.json")


def resolve_policy_output_root() -> str:
    """Training output root (job dirs are `<root>/<job.name>/`)."""
    override = os.environ.get("COSMOS_POLICY_OUTPUT_ROOT")
    if override:
        return os.path.abspath(override)
    return default_checkpoints_root()


def apply_project_default_env() -> None:
    """Apply repo-relative defaults via ``os.environ.setdefault`` (safe to call repeatedly)."""
    global _APPLIED
    root = repo_root()
    hf_home = default_hf_home()
    checkpoints_root = default_checkpoints_root()

    os.environ.setdefault("BASE_DATASETS_DIR", root)
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_home)
    os.environ.setdefault("COSMOS_POLICY_CHECKPOINTS_LAYOUT", "1")
    os.environ.setdefault("COSMOS_POLICY_OUTPUT_ROOT", checkpoints_root)

    os.makedirs(hf_home, exist_ok=True)
    os.makedirs(checkpoints_root, exist_ok=True)
    _APPLIED = True


# Apply as soon as this module is imported (covers config import before train.main).
apply_project_default_env()
