#!/usr/bin/env python3
"""
Minimal check: rotvec (axis-angle) -> quaternion in convert_vla4desk_to_libero_hdf5.py
vs the same conversion path used in vla4desk (scipy.spatial.transform.Rotation).

vla4desk stores proprio as 8-dim state with rotvec in state[3:6]; it does not write
9-dim quat proprio to disk. This script compares the convert mapping to scipy calls
as used in vla4desk (e.g. franka_env, data_recorder).

Usage:
    python scripts/compare_rotvec_quat_vla4desk.py
    python scripts/compare_rotvec_quat_vla4desk.py --json /path/to/epo_1/data.json --max-samples 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cosmos_policy.config.output_paths import default_vla4desk_example_data_json  # noqa: E402
_VLA4DESK_SRC = _PROJECT_ROOT.parent / "vla4desk" / "src"
if _VLA4DESK_SRC.is_dir() and str(_VLA4DESK_SRC) not in sys.path:
    sys.path.insert(0, str(_VLA4DESK_SRC))


def state_to_robot_state_convert(state: np.ndarray) -> np.ndarray:
    """Same as scripts/convert_vla4desk_to_libero_hdf5.py::state_to_robot_state."""
    state = np.asarray(state, dtype=np.float32)
    gripper_qpos = np.array([state[6], state[7]], dtype=np.float32)
    eef_pos = state[0:3]
    eef_quat = Rotation.from_rotvec(state[3:6]).as_quat().astype(np.float32)
    return np.concatenate([gripper_qpos, eef_pos, eef_quat], axis=0)


def rotvec_to_quat_vla4desk(rotvec: np.ndarray) -> np.ndarray:
    """scipy path used throughout vla4desk (franka_env / data_recorder)."""
    return Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_quat()


def rotvec_to_quat_via_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Compose via matrix (as in franka_env _apply_delta_to_pose_array)."""
    r = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64))
    return Rotation.from_matrix(r.as_matrix()).as_quat()


def quat_xyzw_angular_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angle (rad) between two unit quaternions; treats q and -q as same."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.abs(np.dot(q1, q2)))
    dot = min(1.0, dot)
    return 2.0 * np.arccos(dot)


def load_sample_rotvecs(json_path: Path | None, max_samples: int) -> list[np.ndarray]:
    if json_path is None or not json_path.is_file():
        rng = np.random.default_rng(0)
        out = []
        for _ in range(max_samples):
            rotvec = rng.normal(size=3)
            rotvec *= rng.uniform(0.0, np.pi)
            out.append(rotvec.astype(np.float64))
        return out

    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)
    frames = meta.get("frames", [])[:max_samples]
    return [np.asarray(fr["state"], dtype=np.float64)[3:6] for fr in frames]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(default_vla4desk_example_data_json()),
        help="Optional data.json; uses random rotvecs if missing",
    )
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-5, help="Max |q_convert - q_ref| (after float32)")
    args = parser.parse_args()

    rotvecs = load_sample_rotvecs(args.json if args.json.is_file() else None, args.max_samples)
    print(f"Samples: {len(rotvecs)}")
    if args.json.is_file():
        print(f"Source: {args.json}")
    else:
        print("Source: random rotvecs (no json)")

    max_err_vla = 0.0
    max_ang_vla = 0.0
    max_ang_mat = 0.0
    max_rt_rot = 0.0

    for rotvec in rotvecs:
        state8 = np.zeros(8, dtype=np.float32)
        state8[3:6] = rotvec.astype(np.float32)
        # robot_states layout: gripper(2) + pos(3) + quat(4)
        q_convert = state_to_robot_state_convert(state8)[5:9].astype(np.float64)

        q_vla = rotvec_to_quat_vla4desk(rotvec)
        q_mat = rotvec_to_quat_via_matrix(rotvec)

        max_err_vla = max(max_err_vla, float(np.max(np.abs(q_convert - q_vla))))
        max_ang_vla = max(max_ang_vla, quat_xyzw_angular_distance(q_convert, q_vla))
        max_ang_mat = max(max_ang_mat, quat_xyzw_angular_distance(q_convert, q_mat))

        r0 = Rotation.from_rotvec(rotvec)
        r1 = Rotation.from_quat(q_convert)
        max_rt_rot = max(max_rt_rot, float(np.max(np.abs(r0.as_matrix() - r1.as_matrix()))))

    print()
    print("convert state[3:6] -> quat  vs  vla4desk Rotation.from_rotvec().as_quat()")
    print(f"  max |Δq| (element):     {max_err_vla:.3e}  (tol {args.tol})")
    print(f"  max angular dist (rad): {max_ang_vla:.3e}")
    print("convert vs via-matrix (franka_env composition path; compare by angle only)")
    print(f"  max angular dist (rad): {max_ang_mat:.3e}")
    print(f"  max |R_rt - R| (matrix): {max_rt_rot:.3e}  (quat->rotvec sign ambiguity ignored)")
    print()
    print("Quaternion order: scipy as_quat() -> [x, y, z, w] (xyzw), matches LIBERO eef_quat layout.")

    ok = max_err_vla <= args.tol and max_ang_vla <= 1e-6 and max_ang_mat <= 1e-6 and max_rt_rot <= 1e-5
    if ok:
        print("PASS: convert and vla4desk scipy conversions match on all samples.")
    else:
        print("FAIL: see metrics above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
