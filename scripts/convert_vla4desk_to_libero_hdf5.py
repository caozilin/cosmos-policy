#!/usr/bin/env python3
"""
Convert VLA4Desk collected episodes (MP4 + data.json) to Cosmos Policy LIBERO-style HDF5.

Source layout (vla4desk/collected):
    <task_name>/epo_<N>/
        cam1.mp4          # third-person (external D435)
        cam2.mp4          # wrist (eye-in-hand D435)
        data.json

Target layout (LIBERO demonstration HDF5, compatible with LIBERODataset):
    <output_dir>/<suite_name>/
        <prompt_slug>_demo.hdf5
        prompts.txt              # one instruction per line (always written)
        t5_embeddings.pkl        # optional: prompts that exist in source pkl
        conversion_manifest.json

HDF5 per file:
    attrs[task_description] = exact prompt string
    data/demo_<k>/
        actions          (T, 7) float32  — same values as data.json (no action_scale undo)
        robot_states     (T, 9) float32
        obs/agentview_rgb[_jpeg], obs/eye_in_hand_rgb[_jpeg]

LIBERODataset still derives `command` from the filename; use prompt_to_hdf5_basename() for a
round-trip match. Point training at <output>/<suite_name>/t5_embeddings.pkl when present.

Example:
    python scripts/convert_vla4desk_to_libero_hdf5.py \\
        --input /media/czl/sata/franka_my_code/vla4desk/collected \\
        --output ./datasets/VLA4Desk-Franka/success_only
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from cosmos_policy.utils.utils import jpeg_encode_image

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_T5_CANDIDATES = (
    _PROJECT_ROOT / "vla4desk" / "t5_embeddings.pkl",
)


def prompt_to_hdf5_basename(prompt: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        raise ValueError("Empty prompt cannot be converted to HDF5 basename.")
    return f"{slug}_demo.hdf5"


def hdf5_basename_to_prompt(basename: str) -> str:
    if not basename.endswith("_demo.hdf5"):
        raise ValueError(f"Unexpected HDF5 basename: {basename}")
    words = basename[:-10].split("_")
    command = ""
    for w in words:
        if "SCENE" in w:
            command = ""
            continue
        command = command + w + " "
    return command[:-1]


def state_to_robot_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (8,):
        raise ValueError(f"Expected state shape (8,), got {state.shape}")
    gripper_qpos = np.array([state[6], state[7]], dtype=np.float32)
    eef_pos = state[0:3]
    eef_quat = Rotation.from_rotvec(state[3:6]).as_quat().astype(np.float32)
    return np.concatenate([gripper_qpos, eef_pos, eef_quat], axis=0)


def read_video_rgb(path: Path, max_frames: int | None = None) -> np.ndarray:
    frames = []
    for idx, frame in enumerate(iio.imiter(str(path))):
        if max_frames is not None and idx >= max_frames:
            break
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 4:
            frame = frame[..., :3]
        frames.append(np.ascontiguousarray(frame, dtype=np.uint8))
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return np.stack(frames, axis=0)


def resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    out = np.empty((len(frames), size, size, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        out[i] = np.asarray(Image.fromarray(frame).resize((size, size), Image.BILINEAR))
    return out


def _episode_sort_key(ep_dir: Path) -> int:
    suffix = ep_dir.name.split("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def discover_episodes(input_dir: Path) -> list[Path]:
    episodes = []
    for task_dir in sorted(input_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        ep_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("epo_")]
        for ep_dir in sorted(ep_dirs, key=_episode_sort_key):
            episodes.append(ep_dir)
    return episodes


def load_episode(ep_dir: Path, resize: int | None) -> dict:
    json_path = ep_dir / "data.json"
    cam1_path = ep_dir / "cam1.mp4"
    cam2_path = ep_dir / "cam2.mp4"
    for p in (json_path, cam1_path, cam2_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)

    prompt = meta.get("prompt", "").strip()
    if not prompt:
        raise ValueError(f"Empty prompt in {json_path}")

    frames_meta = meta["frames"]
    if not frames_meta:
        raise ValueError(f"No frames in {json_path}")

    num_frames = int(meta.get("num_frames", len(frames_meta)))
    agentview = read_video_rgb(cam1_path, max_frames=num_frames)
    wrist = read_video_rgb(cam2_path, max_frames=num_frames)

    t_use = min(len(frames_meta), len(agentview), len(wrist))
    frames_meta = frames_meta[:t_use]
    agentview = agentview[:t_use]
    wrist = wrist[:t_use]

    actions = np.stack(
        [np.asarray(frame["action"], dtype=np.float32) for frame in frames_meta],
        axis=0,
    )
    robot_states = np.stack(
        [state_to_robot_state(frame["state"]) for frame in frames_meta],
        axis=0,
    )

    if resize is not None and resize > 0:
        agentview = resize_frames(agentview, resize)
        wrist = resize_frames(wrist, resize)

    return {
        "prompt": prompt,
        "task_name": meta.get("task_name", ep_dir.parent.name),
        "episode_dir": str(ep_dir),
        "actions": actions,
        "robot_states": robot_states,
        "agentview_rgb": agentview,
        "eye_in_hand_rgb": wrist,
    }


def write_demo_group(
    h5_group: h5py.Group,
    demo_key: str,
    episode: dict,
    jpeg_compress: bool,
) -> None:
    ep_grp = h5_group.create_group(demo_key)
    obs_grp = ep_grp.create_group("obs")

    agentview = episode["agentview_rgb"]
    wrist = episode["eye_in_hand_rgb"]

    if jpeg_compress:
        dt = h5py.vlen_dtype(np.dtype("uint8"))
        obs_grp.create_dataset(
            "agentview_rgb_jpeg",
            data=[jpeg_encode_image(f) for f in agentview],
            dtype=dt,
        )
        obs_grp.create_dataset(
            "eye_in_hand_rgb_jpeg",
            data=[jpeg_encode_image(f) for f in wrist],
            dtype=dt,
        )
    else:
        obs_grp.create_dataset("agentview_rgb", data=agentview, compression="gzip", compression_opts=4)
        obs_grp.create_dataset("eye_in_hand_rgb", data=wrist, compression="gzip", compression_opts=4)

    ep_grp.create_dataset("actions", data=episode["actions"], compression="gzip", compression_opts=4)
    ep_grp.create_dataset("robot_states", data=episode["robot_states"], compression="gzip", compression_opts=4)


def resolve_t5_source(path: str | None) -> Path | None:
    if path:
        candidate = Path(path).expanduser().resolve()
        return candidate if candidate.is_file() else None
    for candidate in _DEFAULT_T5_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def write_prompt_artifacts(
    suite_dir: Path,
    prompts: list[str],
    t5_source: Path | None,
    *,
    copy_full_t5: bool,
) -> dict:
    """Write prompts.txt and optional t5_embeddings.pkl (subset or full copy)."""
    prompts_path = suite_dir / "prompts.txt"
    with open(prompts_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(prompt + "\n")

    info: dict = {
        "prompts_txt": str(prompts_path),
        "prompts_count": len(prompts),
        "t5_embeddings_pkl": None,
        "t5_source": str(t5_source) if t5_source else None,
        "prompts_with_t5": [],
        "prompts_without_t5": list(prompts),
    }

    if t5_source is None:
        return info

    with open(t5_source, "rb") as f:
        source_embeddings = pickle.load(f)

    if copy_full_t5:
        out_embeddings = source_embeddings
        with_t5 = sorted(source_embeddings.keys())
        without_t5 = sorted(set(prompts) - set(with_t5))
    else:
        with_t5 = [p for p in prompts if p in source_embeddings]
        without_t5 = [p for p in prompts if p not in source_embeddings]
        out_embeddings = {p: source_embeddings[p] for p in with_t5}

    if out_embeddings:
        t5_out = suite_dir / "t5_embeddings.pkl"
        with open(t5_out, "wb") as f:
            pickle.dump(out_embeddings, f)
        info["t5_embeddings_pkl"] = str(t5_out)
        info["t5_embeddings_count"] = len(out_embeddings)

    info["prompts_with_t5"] = with_t5
    info["prompts_without_t5"] = without_t5
    return info


class _PromptH5Writer:
    """One open HDF5 file per prompt; demos are appended incrementally."""

    def __init__(self, suite_dir: Path, *, append: bool, jpeg_compress: bool):
        self.suite_dir = suite_dir
        self.append = append
        self.jpeg_compress = jpeg_compress
        self._open: dict[str, tuple[h5py.File, h5py.Group, int]] = {}
        self.demo_counts: dict[str, int] = defaultdict(int)

    def write_episode(self, episode: dict) -> str:
        prompt = episode["prompt"]
        basename = prompt_to_hdf5_basename(prompt)
        recovered = hdf5_basename_to_prompt(basename)
        if recovered != prompt:
            raise RuntimeError(
                f"Prompt round-trip failed: {prompt!r} -> {basename} -> {recovered!r}"
            )

        if prompt not in self._open:
            out_path = self.suite_dir / basename
            mode = "a" if self.append and out_path.exists() else "w"
            h5f = h5py.File(out_path, mode)
            h5f.attrs["task_description"] = prompt
            if "data" not in h5f:
                data_grp = h5f.create_group("data")
            else:
                data_grp = h5f["data"]
            existing = [k for k in data_grp.keys() if k.startswith("demo_")]
            next_idx = max((int(k.split("_")[1]) for k in existing), default=-1) + 1
            self._open[prompt] = (h5f, data_grp, next_idx)

        h5f, data_grp, next_idx = self._open[prompt]
        write_demo_group(data_grp, f"demo_{next_idx}", episode, jpeg_compress=self.jpeg_compress)
        self._open[prompt] = (h5f, data_grp, next_idx + 1)
        self.demo_counts[prompt] += 1
        return prompt

    def close(self) -> dict[str, dict]:
        files: dict[str, dict] = {}
        for prompt, (h5f, _, _) in self._open.items():
            basename = prompt_to_hdf5_basename(prompt)
            files[prompt] = {
                "hdf5_path": str(self.suite_dir / basename),
                "basename": basename,
                "task_description": prompt,
                "num_demos": self.demo_counts[prompt],
            }
            h5f.close()
        self._open.clear()
        return files


def convert_dataset(args: argparse.Namespace) -> None:
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    suite_dir = output_dir / args.suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    errors: list[dict[str, str]] = []
    prompts_seen: set[str] = set()

    episode_dirs = discover_episodes(input_dir)
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]

    print(
        f"Converting {len(episode_dirs)} episodes (no T5 model; "
        f"resize={args.resize}, jpeg={args.jpeg_compress})..."
    )
    sys.stdout.flush()

    writer = _PromptH5Writer(suite_dir, append=args.append, jpeg_compress=args.jpeg_compress)
    for ep_dir in tqdm(episode_dirs, desc="Episodes", unit="ep"):
        try:
            episode = load_episode(ep_dir, resize=args.resize)
            prompt = writer.write_episode(episode)
            prompts_seen.add(prompt)
            del episode
        except Exception as exc:  # noqa: BLE001
            errors.append({"episode": str(ep_dir), "error": str(exc)})

    manifest: dict = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "suite_name": args.suite_name,
        "files": writer.close(),
        "errors": errors,
    }

    all_prompts = sorted(prompts_seen)
    t5_source = None if args.skip_t5 else resolve_t5_source(args.t5_embeddings)
    if t5_source:
        print(f"Using T5 embeddings source: {t5_source}")
    else:
        print("No T5 embeddings source found; writing prompts.txt only.")

    manifest["language"] = write_prompt_artifacts(
        suite_dir,
        all_prompts,
        t5_source,
        copy_full_t5=args.copy_full_t5,
    )

    manifest_path = suite_dir / "conversion_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    lang = manifest["language"]
    print(f"Wrote {len(manifest['files'])} HDF5 file(s) under {suite_dir}")
    print(f"prompts.txt: {lang['prompts_count']} prompts")
    if lang.get("t5_embeddings_pkl"):
        print(
            f"t5_embeddings.pkl: {lang['t5_embeddings_count']} entries "
            f"({len(lang['prompts_without_t5'])} prompts still need precompute)"
        )
    else:
        print("t5_embeddings.pkl: not written (run scripts/precompute_t5_embeddings.py on prompts.txt)")
    print(f"Manifest: {manifest_path}")
    if errors:
        print(f"Errors: {len(errors)}")
        for item in errors[:5]:
            print(f"  {item['episode']}: {item['error']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VLA4Desk collected data to LIBERO-style HDF5")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="/media/czl/sata/franka_my_code/vla4desk/collected",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./datasets/VLA4Desk-Franka/success_only",
    )
    parser.add_argument(
        "--suite-name",
        type=str,
        default="vla4desk_franka",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=256,
        help="Resize frames to NxN (0 = keep native resolution)",
    )
    parser.add_argument(
        "--jpeg-compress",
        action="store_true",
    )
    parser.add_argument(
        "--skip-t5",
        action="store_true",
        help="Do not read or copy t5_embeddings.pkl (only write prompts.txt)",
    )
    parser.add_argument(
        "--t5-embeddings",
        type=str,
        default=None,
        help="Source t5_embeddings.pkl (default: vla4desk/t5_embeddings.pkl if present)",
    )
    parser.add_argument(
        "--copy-full-t5",
        action="store_true",
        help="Copy entire source t5_embeddings.pkl instead of only converted prompts",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append demos to existing HDF5 files",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert at most N episodes (for testing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resize == 0:
        args.resize = None
    convert_dataset(args)


if __name__ == "__main__":
    main()
