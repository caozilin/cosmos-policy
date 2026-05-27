#!/usr/bin/env python3
"""
Serve VLA4Desk Franka LoRA Cosmos Policy over the OpenPI WebSocket protocol.

vla4desk Coordinator + WebsocketClientPolicy connect unchanged; this server converts
openpi-shaped observations to Cosmos inputs and returns {"actions": (H, 7)}.

Example (GPU machine, repo root):
  uv run --extra cu128 --python 3.10 python scripts/serve_vla4desk_openpi_ws.py \\
    --lora-run my_franka_lora_1gpu_3 \\
    --port 8000
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import socket
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cosmos_policy.config.output_paths import (  # noqa: E402
    apply_project_default_env,
    default_vla4desk_franka_dataset_dir,
)

apply_project_default_env()

from cosmos_policy.serving.vla4desk_openpi_policy import (  # noqa: E402
    Vla4deskOpenPIBridgePolicy,
    Vla4deskServeConfig,
)
from cosmos_policy.serving.websocket_policy_server import WebsocketPolicyServer  # noqa: E402

_DEFAULT_LORA_RUN = "my_franka_lora_1gpu_3"
_DEFAULT_CONFIG = "cosmos_predict2_2b_480p_vla4desk_franka_135_demos_lora__inference_only"
_DEFAULT_DATASET_DIR = Path(default_vla4desk_franka_dataset_dir())


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


def resolve_dataset_stats_path(dataset_stats_path: str) -> str:
    if dataset_stats_path:
        p = Path(dataset_stats_path).expanduser().resolve()
        if p.is_file():
            return str(p)
    default = _DEFAULT_DATASET_DIR / "dataset_statistics.json"
    if default.is_file():
        return str(default.resolve())
    raise FileNotFoundError(
        f"dataset_statistics.json not found. Pass --dataset-stats-path (tried {default})"
    )


def resolve_t5_path(t5_path: str) -> str:
    if t5_path:
        p = Path(t5_path).expanduser().resolve()
        if p.is_file():
            return str(p)
    for candidate in (
        _DEFAULT_DATASET_DIR / "t5_embeddings.pkl",
        _PROJECT_ROOT / "vla4desk" / "t5_embeddings.pkl",
    ):
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(
        "t5_embeddings.pkl not found. Pass --t5-text-embeddings-path or run precompute_t5_embeddings.py"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenPI WebSocket server for VLA4Desk Franka Cosmos LoRA")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", default=_DEFAULT_CONFIG)
    p.add_argument("--lora-run", default=_DEFAULT_LORA_RUN)
    p.add_argument("--ckpt-path", type=Path, default=None, help="Override checkpoint root or iter_*/model")
    p.add_argument("--dataset-stats-path", default="")
    p.add_argument("--t5-text-embeddings-path", default="")
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--flip-images", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()

    os.environ["DETERMINISTIC"] = "True"

    ckpt_root = (
        args.ckpt_path.expanduser().resolve()
        if args.ckpt_path is not None
        else (_PROJECT_ROOT / "checkpoints" / args.lora_run / "checkpoints").resolve()
    )
    ckpt_path = resolve_finetuned_checkpoint_path(ckpt_root)
    dataset_stats_path = resolve_dataset_stats_path(args.dataset_stats_path)
    t5_path = resolve_t5_path(args.t5_text_embeddings_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = Vla4deskServeConfig(
        config=args.config,
        ckpt_path=str(ckpt_path),
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=dataset_stats_path,
        t5_text_embeddings_path=t5_path,
        flip_images=args.flip_images,
        seed=args.seed,
    )

    logging.info("LoRA run: %s", args.lora_run)
    logging.info("Checkpoint: %s", ckpt_path)
    logging.info("Dataset stats: %s", dataset_stats_path)
    logging.info("T5 embeddings: %s", t5_path)

    policy = Vla4deskOpenPIBridgePolicy(cfg)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Starting OpenPI bridge server (host=%s ip=%s port=%s)", hostname, local_ip, args.port)

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
