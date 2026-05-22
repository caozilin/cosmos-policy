#!/usr/bin/env python3
"""
Run LIBERO simulation evaluation with the VLA4Desk Franka LoRA fine-tuned checkpoint.

Example:
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  uv run --extra cu128 --group libero --python 3.10 \
    python scripts/eval_vla4desk_finetuned_lora_libero.py \
    --task_suite_name libero_spatial \
    --num_trials_per_task 1
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cosmos_policy.config.output_paths import (  # noqa: E402
    apply_project_default_env,
    default_vla4desk_franka_dataset_dir,
)

apply_project_default_env()

from cosmos_policy.experiments.robot.libero.run_libero_eval import eval_libero  # noqa: E402

_DEFAULT_CKPT_ROOT = _PROJECT_ROOT / "checkpoints/my_franka_lora_1gpu_1/checkpoints"
_DEFAULT_DATASET_STATS = Path(default_vla4desk_franka_dataset_dir()) / "dataset_statistics.json"
_DEFAULT_T5 = Path(default_vla4desk_franka_dataset_dir()) / "t5_embeddings.pkl"


def _iter_number(path: Path) -> int:
    match = re.search(r"iter_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def resolve_finetuned_checkpoint_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if (path / "model").is_dir() and path.name.startswith("iter_"):
        return path / "model"
    latest_txt = path / "latest_checkpoint.txt"
    if latest_txt.is_file():
        latest = latest_txt.read_text(encoding="utf-8").strip()
        if latest:
            latest_path = Path(latest)
            if not latest_path.is_absolute():
                latest_path = path / latest_path
            if latest_path.is_dir() and (latest_path / "model").is_dir():
                return latest_path / "model"
            if latest_path.is_dir() and latest_path.name == "model":
                return latest_path
            return latest_path
    model_dirs = sorted(
        (candidate / "model" for candidate in path.glob("iter_*") if (candidate / "model").is_dir()),
        key=lambda p: _iter_number(p.parent),
    )
    if model_dirs:
        return model_dirs[-1]
    return path


def main() -> None:
    argv = sys.argv[1:]

    if "--config" not in argv:
        argv.extend(["--config", "cosmos_predict2_2b_480p_vla4desk_franka_135_demos_lora__inference_only"])
    if "--ckpt_path" not in argv and "--ckpt-path" not in argv:
        argv.extend(["--ckpt_path", str(resolve_finetuned_checkpoint_path(_DEFAULT_CKPT_ROOT))])
    if "--dataset_stats_path" not in argv and "--dataset-stats-path" not in argv:
        argv.extend(["--dataset_stats_path", str(_DEFAULT_DATASET_STATS)])
    if "--t5_text_embeddings_path" not in argv and "--t5-text-embeddings-path" not in argv:
        argv.extend(["--t5_text_embeddings_path", str(_DEFAULT_T5)])
    if "--flip_images" not in argv and "--flip-images" not in argv:
        argv.extend(["--flip_images", "True"])
    if "--trained_with_image_aug" not in argv and "--trained-with-image-aug" not in argv:
        argv.extend(["--trained_with_image_aug", "True"])
    if "--num_denoising_steps_action" not in argv and "--num-denoising-steps-action" not in argv:
        argv.extend(["--num_denoising_steps_action", "5"])
    if "--num_denoising_steps_future_state" not in argv and "--num-denoising-steps-future-state" not in argv:
        argv.extend(["--num_denoising_steps_future_state", "1"])
    if "--num_denoising_steps_value" not in argv and "--num-denoising-steps-value" not in argv:
        argv.extend(["--num_denoising_steps_value", "1"])
    if "--run_id_note" not in argv and "--run-id-note" not in argv:
        argv.extend(["--run_id_note", "vla4desk_finetuned_lora"])

    sys.argv = [sys.argv[0], *argv]
    eval_libero()


if __name__ == "__main__":
    main()
