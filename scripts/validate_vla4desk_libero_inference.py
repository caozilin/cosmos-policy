#!/usr/bin/env python3
"""
Offline validation: run LIBERO Cosmos Policy on VLA4Desk data loaded from LIBERO-style HDF5.

All model inputs (images, proprio, ground-truth actions) are read from the converted HDF5
(same layout as LIBERODataset). Timestamps for 0–18 s sampling come only from an optional
data.json sidecar (--timestamps-json); HDF5 does not store per-frame time.

Example (paths default to <repo>/datasets/... and <repo>/vla4desk/collected/...):
  uv run --extra cu128 --group libero --python 3.10 \\
    python scripts/validate_vla4desk_libero_inference.py \\
    --output-dir ./validation_outputs/vla4desk_epo_1_hdf5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cosmos_policy.config.output_paths import (  # noqa: E402
    apply_project_default_env,
    default_hf_home,
    default_vla4desk_example_data_json,
    default_vla4desk_franka_dataset_dir,
    repo_root,
)

apply_project_default_env()

from cosmos_policy.constants import PROPRIO_DIM  # noqa: E402
from cosmos_policy.datasets.dataset_utils import decode_jpeg_bytes_dataset  # noqa: E402
from cosmos_policy.experiments.robot.cosmos_utils import (  # noqa: E402
    extract_action_chunk_from_latent_sequence,
    get_action,
    get_model,
    get_t5_embedding_from_cache,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    rescale_proprio,
    t5_text_embeddings_cache,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig  # noqa: E402
from cosmos_policy.experiments.robot.robot_utils import DATE_TIME  # noqa: E402

_DEFAULT_HDF5 = (
    Path(default_vla4desk_franka_dataset_dir()) / "put_the_yellow_cube_on_the_red_plate_demo.hdf5"
)
_DEFAULT_VLA4DESK_T5 = _DEFAULT_HDF5.parent / "t5_embeddings.pkl"

# robot_states in HDF5: gripper_qpos(2) + eef_pos(3) + eef_quat_xyzw(4)
PROPRIO_NAMES = (
    "gripper_qpos_0",
    "gripper_qpos_1",
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_qx",
    "eef_qy",
    "eef_qz",
    "eef_qw",
)
ACTION_NAMES = ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")


def load_demo_from_hdf5(hdf5_path: Path, demo_key: str = "demo_0") -> dict:
    """Load one demo from LIBERO-style VLA4Desk HDF5 (same paths as LIBERODataset)."""
    with h5py.File(hdf5_path, "r") as f:
        task_description = f.attrs.get("task_description", "")
        if isinstance(task_description, bytes):
            task_description = task_description.decode("utf-8")

        demo_grp = f[f"data/{demo_key}"]
        obs_grp = demo_grp["obs"]

        if "agentview_rgb" in obs_grp:
            agentview = obs_grp["agentview_rgb"][:]
        elif "agentview_rgb_jpeg" in obs_grp:
            agentview = decode_jpeg_bytes_dataset(obs_grp["agentview_rgb_jpeg"])
        else:
            raise KeyError("Neither agentview_rgb nor agentview_rgb_jpeg in HDF5")

        if "eye_in_hand_rgb" in obs_grp:
            wrist = obs_grp["eye_in_hand_rgb"][:]
        elif "eye_in_hand_rgb_jpeg" in obs_grp:
            wrist = decode_jpeg_bytes_dataset(obs_grp["eye_in_hand_rgb_jpeg"])
        else:
            raise KeyError("Neither eye_in_hand_rgb nor eye_in_hand_rgb_jpeg in HDF5")

        actions = demo_grp["actions"][:].astype(np.float32)
        robot_states = demo_grp["robot_states"][:].astype(np.float32)

    return {
        "prompt": str(task_description).strip(),
        "agentview_rgb": np.asarray(agentview, dtype=np.uint8),
        "eye_in_hand_rgb": np.asarray(wrist, dtype=np.uint8),
        "robot_states": robot_states,
        "actions": actions,
        "num_frames": len(actions),
    }


def load_frame_timestamps(timestamps_json: Path | None) -> list[float] | None:
    if timestamps_json is None or not timestamps_json.is_file():
        return None
    with open(timestamps_json, encoding="utf-8") as f:
        meta = json.load(f)
    return [float(frame["timestamp"]) for frame in meta["frames"]]


def frame_index_at_time(timestamps: list[float], target_s: float) -> int:
    return int(min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - target_s)))


def frame_index_uniform(target_s: float, num_frames: int, collect_hz: float) -> int:
    return int(min(max(0, round(target_s * collect_hz)), num_frames - 1))


def resize_uint8(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def make_comparison_image(
    primary_in: np.ndarray,
    wrist_in: np.ndarray,
    future_primary: np.ndarray | None,
    future_wrist: np.ndarray | None,
    tile_size: int = 224,
    title: str = "",
) -> Image.Image:
    panels = [
        ("in: third-person", primary_in),
        ("in: wrist", wrist_in),
        (
            "pred: future third-person",
            future_primary if future_primary is not None else np.zeros((tile_size, tile_size, 3), dtype=np.uint8),
        ),
        (
            "pred: future wrist",
            future_wrist if future_wrist is not None else np.zeros((tile_size, tile_size, 3), dtype=np.uint8),
        ),
    ]
    tiles = []
    for label, arr in panels:
        tile = resize_uint8(np.asarray(arr, dtype=np.uint8), tile_size)
        pil = Image.fromarray(tile)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.rectangle((0, 0, tile_size, 14), fill=(0, 0, 0))
        draw.text((4, 1), label, fill=(255, 255, 255), font=font)
        tiles.append(pil)

    row1 = np.concatenate([np.asarray(tiles[0]), np.asarray(tiles[1])], axis=1)
    row2 = np.concatenate([np.asarray(tiles[2]), np.asarray(tiles[3])], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    out = Image.fromarray(grid)
    if title:
        banner_h = 18
        canvas = Image.new("RGB", (out.width, out.height + banner_h), (30, 30, 30))
        canvas.paste(out, (0, banner_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 2), title, fill=(255, 255, 255))
        out = canvas
    return out


def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    return x


def _libero_cache_roots() -> list[Path]:
    roots: list[Path] = [Path(default_hf_home()).resolve()]
    hf_hub_cache = os.environ.get("HF_HUB_CACHE", "").strip()
    if hf_hub_cache:
        hub_path = Path(hf_hub_cache).expanduser().resolve()
        if hub_path not in roots:
            roots.append(hub_path)
    return roots


def find_libero_snapshot_dir() -> Path | None:
    """Locate cached LIBERO snapshot under HF_HOME and/or project hf_cache."""
    for root in _libero_cache_roots():
        for rel in (
            "models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/snapshots",
            "hub/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/snapshots",
        ):
            snap_parent = root / rel
            if not snap_parent.is_dir():
                continue
            snapshots = sorted(p for p in snap_parent.iterdir() if p.is_dir())
            if snapshots:
                return snapshots[-1]
    return None


def _looks_like_hf_repo_path(path: str) -> bool:
    return bool(path) and not path.startswith(("/", "./")) and "/" in path


def resolve_local_libero_paths(ckpt_path: str, dataset_stats_path: str) -> tuple[str, str, Path | None]:
    """Prefer on-disk LIBERO checkpoint/stats (HF snapshot) to avoid network download."""
    snap = find_libero_snapshot_dir()
    if snap is None:
        return ckpt_path, dataset_stats_path, None

    local_ckpt = snap / "Cosmos-Policy-LIBERO-Predict2-2B.pt"
    local_stats = snap / "libero_dataset_statistics.json"

    if _looks_like_hf_repo_path(ckpt_path) and local_ckpt.is_file():
        ckpt_path = str(local_ckpt)
    if _looks_like_hf_repo_path(dataset_stats_path) and local_stats.is_file():
        dataset_stats_path = str(local_stats)

    return ckpt_path, dataset_stats_path, snap


def resolve_vla4desk_t5_path(hdf5_path: Path, t5_text_embeddings_path: str) -> str:
    """Use Franka/VLA4Desk t5_embeddings.pkl (not LIBERO libero_t5_embeddings.pkl)."""
    if t5_text_embeddings_path and not _looks_like_hf_repo_path(t5_text_embeddings_path):
        p = Path(t5_text_embeddings_path).expanduser().resolve()
        if p.is_file():
            return str(p)

    candidates = [
        hdf5_path.parent / "t5_embeddings.pkl",
        _DEFAULT_VLA4DESK_T5,
        _PROJECT_ROOT / "vla4desk/t5_embeddings.pkl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    return t5_text_embeddings_path


def unnormalize_proprio(
    proprio_norm: np.ndarray, dataset_stats: dict, scale_multiplier: float = 1.0
) -> np.ndarray:
    """Inverse of rescale_proprio (LIBERO stats, [-1, 1] range)."""
    arr = np.asarray(proprio_norm, dtype=np.float64) / scale_multiplier
    curr_min = np.asarray(dataset_stats["proprio_min"], dtype=np.float64)
    curr_max = np.asarray(dataset_stats["proprio_max"], dtype=np.float64)
    return ((arr + 1.0) / 2.0 * (curr_max - curr_min) + curr_min).astype(np.float32)


def extract_predicted_future_proprio(result: dict) -> np.ndarray | None:
    """Read future proprio from generated latent (same layout as action extraction)."""
    if "generated_latent" not in result or "latent_indices" not in result:
        return None
    future_idx = result["latent_indices"].get("future_proprio_latent_idx", -1)
    if future_idx is None or int(future_idx) < 0:
        return None

    latent = result["generated_latent"]
    device = latent.device
    batch_size = latent.shape[0]
    indices = torch.full((batch_size,), int(future_idx), dtype=torch.int64, device=device)
    chunk = extract_action_chunk_from_latent_sequence(
        latent, action_shape=(1, PROPRIO_DIM), action_indices=indices
    )
    return chunk[0, 0].detach().float().cpu().numpy().astype(np.float32)


def future_frame_index(frame_idx: int, chunk_size: int, num_frames: int) -> int:
    """Match LIBERODataset: future proprio at t + chunk_size (clamped)."""
    return min(frame_idx + chunk_size, num_frames - 1)


def named_vector(values: np.ndarray, names: tuple[str, ...]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {names[i]: float(arr[i]) for i in range(min(len(names), len(arr)))}


def get_dataset_action_chunk(actions: np.ndarray, frame_idx: int, chunk_size: int) -> np.ndarray:
    """Match training: repeat last action if the chunk extends past episode end."""
    end = min(frame_idx + chunk_size, len(actions))
    chunk = np.asarray(actions[frame_idx:end], dtype=np.float32)
    if len(chunk) < chunk_size:
        pad = np.repeat(chunk[-1:], chunk_size - len(chunk), axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk


def write_states_actions_csv(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_time_targets(start_s: float, end_s: float, step_s: float) -> list[float]:
    targets = []
    t = start_s
    while t <= end_s + 1e-6:
        targets.append(round(t, 6))
        t += step_s
    return targets


def parse_args() -> argparse.Namespace:
    snap = find_libero_snapshot_dir()
    default_ckpt = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
    default_stats = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json"
    if snap is not None:
        default_ckpt, default_stats, _ = resolve_local_libero_paths(default_ckpt, default_stats)
    default_t5 = str(_DEFAULT_VLA4DESK_T5) if _DEFAULT_VLA4DESK_T5.is_file() else ""

    p = argparse.ArgumentParser(
        description="Validate VLA4Desk LIBERO-style HDF5 with Cosmos Policy (LIBERO checkpoint)."
    )
    p.add_argument("--hdf5-path", type=Path, default=_DEFAULT_HDF5, help="Converted demonstration HDF5")
    p.add_argument("--demo-key", type=str, default="demo_0", help="Demo group inside HDF5 data/")
    p.add_argument(
        "--timestamps-json",
        type=Path,
        default=Path(default_vla4desk_example_data_json()),
        help="Optional data.json only for mapping target seconds -> frame index (not used for pixels/proprio)",
    )
    p.add_argument(
        "--collect-hz",
        type=float,
        default=10.0,
        help="Fallback frame index = round(t * hz) when --timestamps-json is missing",
    )
    p.add_argument("--output-dir", type=Path, default=Path("./validation_outputs/vla4desk_epo_1_hdf5"))
    p.add_argument("--start-time", type=float, default=0.0)
    p.add_argument("--end-time", type=float, default=18.0)
    p.add_argument("--time-step", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=195)
    p.add_argument(
        "--task-prompt",
        type=str,
        default=None,
        help="Override language prompt (default: HDF5 attrs task_description)",
    )
    p.add_argument("--flip-images", action="store_true", help="Flip images like LIBERO sim (usually off for real robot)")
    p.add_argument("--config", default="cosmos_predict2_2b_480p_libero__inference_only")
    p.add_argument("--ckpt-path", default=default_ckpt)
    p.add_argument("--dataset-stats-path", default=default_stats)
    p.add_argument(
        "--t5-text-embeddings-path",
        default=default_t5,
        help="VLA4Desk t5_embeddings.pkl (default: next to --hdf5-path or datasets/.../vla4desk_franka/)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["DETERMINISTIC"] = "True"

    hdf5_path = args.hdf5_path.expanduser().resolve()
    args.t5_text_embeddings_path = resolve_vla4desk_t5_path(hdf5_path, args.t5_text_embeddings_path)

    args.ckpt_path, args.dataset_stats_path, libero_snap = resolve_local_libero_paths(
        args.ckpt_path, args.dataset_stats_path
    )
    if libero_snap is not None:
        print(f"Using local LIBERO assets (no HuggingFace download): {libero_snap}")
    else:
        print(
            "Warning: local LIBERO snapshot not found under HF_HOME or ./hf_cache; "
            "will contact huggingface.co for nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
        )

    output_dir = args.output_dir.expanduser().resolve()
    comparisons_dir = output_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading demo {args.demo_key!r} from HDF5: {hdf5_path}")
    episode = load_demo_from_hdf5(hdf5_path, demo_key=args.demo_key)
    timestamps = load_frame_timestamps(
        args.timestamps_json.expanduser().resolve() if args.timestamps_json else None
    )
    if timestamps is not None:
        if len(timestamps) != episode["num_frames"]:
            print(
                f"Warning: timestamps ({len(timestamps)}) != HDF5 frames ({episode['num_frames']}); "
                "clipping to min length for index lookup."
            )
            n = min(len(timestamps), episode["num_frames"])
            timestamps = timestamps[:n]
            episode["num_frames"] = n
        time_index_mode = "timestamps_json"
    else:
        time_index_mode = f"uniform_collect_hz={args.collect_hz}"
        print(f"No timestamps JSON; frame index = round(t * {args.collect_hz})")

    cfg = PolicyEvalConfig(
        config=args.config,
        ckpt_path=args.ckpt_path,
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        chunk_size=16,
        num_open_loop_steps=16,
        trained_with_image_aug=True,
        use_jpeg_compression=True,
        flip_images=args.flip_images,
        num_denoising_steps_action=5,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        seed=args.seed,
        deterministic=True,
    )

    task_prompt = (args.task_prompt or episode["prompt"]).strip()
    if not task_prompt:
        raise ValueError("Empty task prompt. Set HDF5 task_description or pass --task-prompt.")

    print("Loading dataset stats and T5 cache ...")
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    if not Path(cfg.t5_text_embeddings_path).is_file():
        raise FileNotFoundError(
            f"VLA4Desk T5 embeddings not found: {cfg.t5_text_embeddings_path}\n"
            "Run scripts/precompute_t5_embeddings.py on prompts.txt first."
        )
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    if task_prompt not in t5_text_embeddings_cache:
        print(f"Prompt not in pkl; will compute T5 online once: {task_prompt!r}")
    else:
        _ = get_t5_embedding_from_cache(task_prompt)
    print(f"Using task prompt: {task_prompt!r}")
    print(f"T5 embeddings: {cfg.t5_text_embeddings_path}")

    print("Loading model (may download checkpoint on first run) ...")
    model, _ = get_model(cfg)

    time_targets = build_time_targets(args.start_time, args.end_time, args.time_step)
    steps_out = []
    csv_rows: list[dict] = []

    for target_s in time_targets:
        if timestamps is not None:
            frame_idx = frame_index_at_time(timestamps, target_s)
            ts = timestamps[frame_idx]
        else:
            frame_idx = frame_index_uniform(target_s, episode["num_frames"], args.collect_hz)
            ts = None

        primary = episode["agentview_rgb"][frame_idx]
        wrist = episode["eye_in_hand_rgb"][frame_idx]
        proprio_hdf5 = episode["robot_states"][frame_idx].astype(np.float32)
        dataset_action = episode["actions"][frame_idx].astype(np.float32)
        dataset_action_chunk = get_dataset_action_chunk(
            episode["actions"], frame_idx, cfg.chunk_size
        )

        if cfg.normalize_proprio:
            proprio_model_input = rescale_proprio(
                proprio_hdf5.copy(), dataset_stats, non_negative_only=False, scale_multiplier=1.0
            )
        else:
            proprio_model_input = proprio_hdf5.copy()

        if cfg.flip_images:
            primary = np.flipud(primary).copy()
            wrist = np.flipud(wrist).copy()

        observation = {
            "primary_image": primary,
            "wrist_image": wrist,
            "proprio": proprio_hdf5,
        }

        ts_str = f"{ts:.3f}s" if ts is not None else "n/a"
        print(f"  Inference @ target={target_s:.0f}s -> HDF5 frame {frame_idx} (ts={ts_str}) ...")
        result = get_action(
            cfg,
            model,
            dataset_stats,
            observation,
            task_prompt,
            seed=cfg.seed + int(target_s),
            num_denoising_steps_action=cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=True,
        )

        future_preds = result.get("future_image_predictions", {})
        future_primary = future_preds.get("future_image")
        future_wrist = future_preds.get("future_wrist_image")

        title = (
            f"t={target_s:.0f}s | hdf5_frame={frame_idx} | ts={ts_str} | "
            f"value={result.get('value_prediction', float('nan')):.3f}"
        )
        comp_img = make_comparison_image(primary, wrist, future_primary, future_wrist, title=title)
        comp_name = f"compare_t{int(target_s):02d}s_frame{frame_idx}.png"
        comp_path = comparisons_dir / comp_name
        comp_img.save(comp_path)

        pred_actions = result["actions"]
        proprio_in_model = result.get("proprio")
        if proprio_in_model is not None:
            proprio_in_model = np.asarray(proprio_in_model, dtype=np.float32).reshape(-1)

        pred_action_first = np.asarray(pred_actions[0], dtype=np.float32) if pred_actions else None

        step_record = {
            "target_time_s": target_s,
            "frame_index": frame_idx,
            "frame_timestamp_s": ts,
            "robot_state_hdf5": proprio_hdf5,
            "robot_state_hdf5_named": named_vector(proprio_hdf5, PROPRIO_NAMES),
            "proprio_model_input": proprio_model_input,
            "proprio_model_input_named": named_vector(proprio_model_input, PROPRIO_NAMES),
            "proprio_in_model_return": proprio_in_model,
            "dataset_action": dataset_action,
            "dataset_action_named": named_vector(dataset_action, ACTION_NAMES),
            "dataset_action_chunk": dataset_action_chunk,
            "predicted_action_chunk": pred_actions,
            "predicted_action_first": pred_action_first,
            "value_prediction": result.get("value_prediction"),
            "comparison_image": str(comp_path),
        }
        steps_out.append(to_jsonable(step_record))

        csv_row = {
            "target_time_s": target_s,
            "frame_index": frame_idx,
            "frame_timestamp_s": ts if ts is not None else "",
            "value_prediction": result.get("value_prediction"),
        }
        for i, name in enumerate(PROPRIO_NAMES):
            csv_row[f"hdf5_{name}"] = float(proprio_hdf5[i])
        for i, name in enumerate(PROPRIO_NAMES):
            csv_row[f"model_{name}"] = float(proprio_model_input[i])
        for i, name in enumerate(ACTION_NAMES):
            csv_row[f"dataset_{name}"] = float(dataset_action[i])
        if pred_action_first is not None:
            for i, name in enumerate(ACTION_NAMES):
                csv_row[f"pred_{name}"] = float(pred_action_first[i])
        csv_rows.append(csv_row)

        print(
            f"    value={step_record['value_prediction']:.4f} "
            f"model_proprio={np.array2string(proprio_model_input, precision=3)} "
            f"dataset_action={np.array2string(dataset_action, precision=3)} "
            f"pred_a0={np.array2string(pred_action_first, precision=3) if pred_action_first is not None else 'n/a'}"
        )

    summary = {
        "data_source": "hdf5",
        "hdf5_path": str(hdf5_path),
        "demo_key": args.demo_key,
        "dataset_prompt": episode["prompt"],
        "num_frames": episode["num_frames"],
        "time_index_mode": time_index_mode,
        "timestamps_json": str(args.timestamps_json) if args.timestamps_json else None,
        "task_prompt": task_prompt,
        "t5_embeddings_path": cfg.t5_text_embeddings_path,
        "note": (
            "LIBERO checkpoint + VLA4Desk task prompt from t5_embeddings.pkl. "
            "timestamps_json only selects frame index per target second."
        ),
        "config": {
            "ckpt_path": cfg.ckpt_path,
            "flip_images": cfg.flip_images,
            "seed": cfg.seed,
        },
        "time_range_s": {"start": args.start_time, "end": args.end_time, "step": args.time_step},
        "proprio_layout": list(PROPRIO_NAMES),
        "action_layout": list(ACTION_NAMES),
        "normalize_proprio": cfg.normalize_proprio,
        "steps": steps_out,
    }

    json_path = output_dir / f"validation_{DATE_TIME}.json"
    csv_path = output_dir / f"states_actions_{DATE_TIME}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_states_actions_csv(csv_rows, csv_path)
    print(f"\nWrote {len(steps_out)} comparisons to {comparisons_dir}")
    print(f"Wrote JSON summary to {json_path}")
    print(f"Wrote state/action table to {csv_path}")


if __name__ == "__main__":
    main()
