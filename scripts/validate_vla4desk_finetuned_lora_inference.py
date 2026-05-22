#!/usr/bin/env python3
"""
Offline validation: run the VLA4Desk Franka LoRA fine-tuned Cosmos Policy on converted HDF5 data.

Example:
  uv run --extra cu128 --python 3.10 \
    python scripts/validate_vla4desk_finetuned_lora_inference.py \
    --output-dir ./validation_outputs/vla4desk_finetuned_lora
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
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
    default_vla4desk_example_data_json,
    default_vla4desk_franka_dataset_dir,
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
from cosmos_policy.experiments.robot.robot_utils import DATE_TIME  # noqa: E402

_DEFAULT_HDF5 = (
    Path(default_vla4desk_franka_dataset_dir()) / "put_the_yellow_cube_on_the_red_plate_demo.hdf5"
)
_DEFAULT_LORA_RUN = "my_franka_lora_1gpu_1"
_DEFAULT_DATASET_STATS = Path(default_vla4desk_franka_dataset_dir()) / "dataset_statistics.json"
_DEFAULT_VLA4DESK_T5 = _DEFAULT_HDF5.parent / "t5_embeddings.pkl"

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


@dataclass
class PolicyEvalConfig:
    config: str
    ckpt_path: str
    config_file: str
    dataset_stats_path: str
    t5_text_embeddings_path: str
    suite: str = "libero"
    model_family: str = "cosmos"
    planning_model_config_name: str = ""
    planning_model_ckpt_path: str = ""
    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    flip_images: bool = False
    use_variance_scale: bool = False
    use_jpeg_compression: bool = True
    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    trained_with_image_aug: bool = True
    chunk_size: int = 16
    num_open_loop_steps: int = 16
    deterministic: bool = True
    deterministic_reset: bool = False
    deterministic_reset_seed: int | None = None
    use_ensemble_future_state_predictions: bool = False
    num_future_state_predictions_in_ensemble: int = 3
    future_state_ensemble_aggregation_scheme: str = "average"
    use_ensemble_value_predictions: bool = False
    num_value_predictions_in_ensemble: int = 5
    value_ensemble_aggregation_scheme: str = "average"
    search_depth: int = 1
    mask_current_state_action_for_value_prediction: bool = False
    mask_future_state_for_qvalue_prediction: bool = False
    num_queries_best_of_n: int = 1
    randomize_seed: bool = False
    parallel_timeout: int = 15
    seed: int = 195


def load_demo_from_hdf5(hdf5_path: Path, demo_key: str = "demo_0") -> dict:
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


def resolve_vla4desk_t5_path(hdf5_path: Path, t5_text_embeddings_path: str) -> str:
    if t5_text_embeddings_path:
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
    arr = np.asarray(proprio_norm, dtype=np.float64) / scale_multiplier
    curr_min = np.asarray(dataset_stats["proprio_min"], dtype=np.float64)
    curr_max = np.asarray(dataset_stats["proprio_max"], dtype=np.float64)
    return ((arr + 1.0) / 2.0 * (curr_max - curr_min) + curr_min).astype(np.float32)


def extract_predicted_future_proprio(result: dict) -> np.ndarray | None:
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
    return min(frame_idx + chunk_size, num_frames - 1)


def named_vector(values: np.ndarray, names: tuple[str, ...]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {names[i]: float(arr[i]) for i in range(min(len(names), len(arr)))}


def get_dataset_action_chunk(actions: np.ndarray, frame_idx: int, chunk_size: int) -> np.ndarray:
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


def resolve_dataset_stats_path(hdf5_path: Path, dataset_stats_path: str) -> str:
    if dataset_stats_path:
        p = Path(dataset_stats_path).expanduser().resolve()
        if p.is_file():
            return str(p)
    candidates = [
        hdf5_path.parent / "dataset_statistics.json",
        _DEFAULT_DATASET_STATS,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return dataset_stats_path


def model_lora_fingerprint(model) -> dict:
    digest = hashlib.sha256()
    stats = {
        "all": {"num_tensors": 0, "numel": 0, "nonzero": 0, "abs_sum": 0.0, "max_abs": 0.0},
        "lora_A": {"num_tensors": 0, "numel": 0, "nonzero": 0, "abs_sum": 0.0, "max_abs": 0.0},
        "lora_B": {"num_tensors": 0, "numel": 0, "nonzero": 0, "abs_sum": 0.0, "max_abs": 0.0},
    }
    for name, tensor in model.state_dict().items():
        name_lower = name.lower()
        if "lora_" not in name_lower:
            continue
        data = tensor.detach().float().cpu().contiguous()
        abs_data = data.abs()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(data.shape)).encode("utf-8"))
        digest.update(data.numpy().tobytes())

        groups = ["all"]
        if "lora_a" in name_lower:
            groups.append("lora_A")
        elif "lora_b" in name_lower:
            groups.append("lora_B")

        for group in groups:
            stats[group]["num_tensors"] += 1
            stats[group]["numel"] += data.numel()
            stats[group]["nonzero"] += int(torch.count_nonzero(data).item())
            stats[group]["abs_sum"] += float(abs_data.sum().item())
            stats[group]["max_abs"] = max(stats[group]["max_abs"], float(abs_data.max().item()))

    net = getattr(model, "net", None)
    active_adapters = getattr(net, "active_adapters", None)
    if active_adapters is None:
        active_adapters = getattr(net, "active_adapter", None)

    return {
        "sha256": digest.hexdigest(),
        "num_tensors": stats["all"]["num_tensors"],
        "numel": stats["all"]["numel"],
        "abs_sum": stats["all"]["abs_sum"],
        "max_abs": stats["all"]["max_abs"],
        "nonzero": stats["all"]["nonzero"],
        "lora_A": stats["lora_A"],
        "lora_B": stats["lora_B"],
        "active_adapters": active_adapters,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate VLA4Desk HDF5 with the Franka LoRA fine-tuned Cosmos Policy checkpoint."
    )
    p.add_argument("--hdf5-path", type=Path, default=_DEFAULT_HDF5, help="Converted demonstration HDF5")
    p.add_argument("--demo-key", type=str, default="demo_0", help="Demo group inside HDF5 data/")
    p.add_argument(
        "--timestamps-json",
        type=Path,
        default=Path(default_vla4desk_example_data_json()),
        help="Optional data.json only for mapping target seconds -> frame index",
    )
    p.add_argument(
        "--collect-hz",
        type=float,
        default=10.0,
        help="Fallback frame index = round(t * hz) when --timestamps-json is missing",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./validation_outputs"),
        help="Root directory; each validation run creates a new timestamped subdirectory under it",
    )
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
    p.add_argument(
        "--config",
        default="cosmos_predict2_2b_480p_vla4desk_franka_135_demos_lora__inference_only",
    )
    p.add_argument(
        "--lora-run",
        default=_DEFAULT_LORA_RUN,
        help="Fine-tuned run name under ./checkpoints, e.g. my_franka_lora_1gpu_1 or my_franka_lora_1gpu_2",
    )
    p.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help="Override fine-tuned checkpoint root, iter dir, model dir, or .pt file",
    )
    p.add_argument(
        "--dataset-stats-path",
        default=str(_DEFAULT_DATASET_STATS) if _DEFAULT_DATASET_STATS.is_file() else "",
        help="VLA4Desk dataset_statistics.json for proprio/action scaling",
    )
    p.add_argument(
        "--t5-text-embeddings-path",
        default="",
        help="VLA4Desk t5_embeddings.pkl (default: next to --hdf5-path or datasets/.../vla4desk_franka/)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["DETERMINISTIC"] = "True"

    hdf5_path = args.hdf5_path.expanduser().resolve()
    ckpt_root = (
        args.ckpt_path.expanduser().resolve()
        if args.ckpt_path is not None
        else (_PROJECT_ROOT / "checkpoints" / args.lora_run / "checkpoints").resolve()
    )
    ckpt_path = resolve_finetuned_checkpoint_path(ckpt_root)
    args.dataset_stats_path = resolve_dataset_stats_path(hdf5_path, args.dataset_stats_path)
    args.t5_text_embeddings_path = resolve_vla4desk_t5_path(hdf5_path, args.t5_text_embeddings_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {ckpt_path}")
    if not args.dataset_stats_path or not Path(args.dataset_stats_path).is_file():
        raise FileNotFoundError(
            "VLA4Desk dataset statistics not found. Expected dataset_statistics.json next to the HDF5 "
            "or pass --dataset-stats-path."
        )
    if not Path(args.t5_text_embeddings_path).is_file():
        raise FileNotFoundError(
            f"VLA4Desk T5 embeddings not found: {args.t5_text_embeddings_path}\n"
            "Run scripts/precompute_t5_embeddings.py on prompts.txt first."
        )

    output_root = args.output_dir.expanduser().resolve()
    output_dir = output_root / f"vla4desk_finetuned_lora_{args.lora_run}_{DATE_TIME}"
    comparisons_dir = output_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using LoRA run: {args.lora_run}")
    print(f"Using checkpoint root: {ckpt_root}")
    print(f"Using fine-tuned checkpoint: {ckpt_path}")
    print(f"Using dataset stats: {args.dataset_stats_path}")
    print(f"Using T5 embeddings: {args.t5_text_embeddings_path}")
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
        ckpt_path=str(ckpt_path),
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
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    if task_prompt not in t5_text_embeddings_cache:
        print(f"Prompt not in pkl; will compute T5 online once: {task_prompt!r}")
    else:
        _ = get_t5_embedding_from_cache(task_prompt)
    print(f"Using task prompt: {task_prompt!r}")

    print("Loading LoRA fine-tuned model ...")
    model, _ = get_model(cfg)
    lora_fingerprint = model_lora_fingerprint(model)
    print(
        "Loaded LoRA fingerprint: "
        f"sha256={lora_fingerprint['sha256']} "
        f"tensors={lora_fingerprint['num_tensors']} "
        f"numel={lora_fingerprint['numel']} "
        f"nonzero={lora_fingerprint['nonzero']} "
        f"abs_sum={lora_fingerprint['abs_sum']:.6f} "
        f"max_abs={lora_fingerprint['max_abs']:.6f} "
        f"lora_A_abs_sum={lora_fingerprint['lora_A']['abs_sum']:.6f} "
        f"lora_B_abs_sum={lora_fingerprint['lora_B']['abs_sum']:.6f} "
        f"active_adapters={lora_fingerprint['active_adapters']}"
    )

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
        dataset_action_chunk = get_dataset_action_chunk(episode["actions"], frame_idx, cfg.chunk_size)

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
        pred_action_first = np.asarray(pred_actions[0], dtype=np.float32) if pred_actions else None
        future_idx = future_frame_index(frame_idx, cfg.chunk_size, episode["num_frames"])
        future_proprio_hdf5 = episode["robot_states"][future_idx].astype(np.float32)
        pred_future_proprio = extract_predicted_future_proprio(result)
        if pred_future_proprio is not None:
            pred_future_proprio = unnormalize_proprio(pred_future_proprio, dataset_stats)

        step_record = {
            "target_time_s": target_s,
            "frame_index": frame_idx,
            "frame_timestamp_s": ts,
            "robot_state_hdf5": proprio_hdf5,
            "robot_state_hdf5_named": named_vector(proprio_hdf5, PROPRIO_NAMES),
            "proprio_model_input": proprio_model_input,
            "proprio_model_input_named": named_vector(proprio_model_input, PROPRIO_NAMES),
            "future_frame_index": future_idx,
            "future_robot_state_hdf5": future_proprio_hdf5,
            "predicted_future_robot_state": pred_future_proprio,
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
            "LoRA fine-tuned VLA4Desk Franka checkpoint + VLA4Desk dataset_statistics.json. "
            "timestamps_json only selects frame index per target second."
        ),
        "config": {
            "config_name": cfg.config,
            "lora_run": args.lora_run,
            "ckpt_root": str(ckpt_root),
            "ckpt_path": cfg.ckpt_path,
            "lora_fingerprint": lora_fingerprint,
            "output_root": str(output_root),
            "output_dir": str(output_dir),
            "dataset_stats_path": cfg.dataset_stats_path,
            "flip_images": cfg.flip_images,
            "seed": cfg.seed,
        },
        "time_range_s": {"start": args.start_time, "end": args.end_time, "step": args.time_step},
        "proprio_layout": list(PROPRIO_NAMES),
        "action_layout": list(ACTION_NAMES),
        "normalize_proprio": cfg.normalize_proprio,
        "steps": steps_out,
    }

    json_path = output_dir / f"validation_finetuned_lora_{DATE_TIME}.json"
    csv_path = output_dir / f"states_actions_finetuned_lora_{DATE_TIME}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_states_actions_csv(csv_rows, csv_path)
    print(f"\nWrote {len(steps_out)} comparisons to {comparisons_dir}")
    print(f"Wrote JSON summary to {json_path}")
    print(f"Wrote state/action table to {csv_path}")


if __name__ == "__main__":
    main()
