"""Bridge OpenPI-shaped observations to VLA4Desk Franka Cosmos Policy inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from cosmos_policy.constants import PROPRIO_DIM
from cosmos_policy.experiments.robot.cosmos_utils import (
    extract_action_chunk_from_latent_sequence,
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.serving.base_policy import BasePolicy

logger = logging.getLogger(__name__)


def state_to_robot_state(state: np.ndarray) -> np.ndarray:
    """Match scripts/convert_vla4desk_to_libero_hdf5.state_to_robot_state."""
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (8,):
        raise ValueError(f"Expected observation/state shape (8,), got {state.shape}")
    gripper_qpos = np.array([state[6], state[7]], dtype=np.float32)
    eef_pos = state[0:3]
    eef_quat = Rotation.from_rotvec(state[3:6]).as_quat().astype(np.float32)
    return np.concatenate([gripper_qpos, eef_pos, eef_quat], axis=0)


def openpi_obs_to_cosmos(obs: dict) -> tuple[dict[str, Any], str]:
    """Convert vla4desk/openpi observation dict to Cosmos get_action input."""
    prompt = str(obs.get("prompt", "")).strip()
    if not prompt:
        raise ValueError('OpenPI observation missing non-empty "prompt"')

    primary = np.asarray(obs["observation/image"], dtype=np.uint8)
    wrist = np.asarray(obs["observation/wrist_image"], dtype=np.uint8)
    state = np.asarray(obs["observation/state"], dtype=np.float64)

    cosmos_obs = {
        "primary_image": primary,
        "wrist_image": wrist,
        "proprio": state_to_robot_state(state),
    }
    return cosmos_obs, prompt


def actions_to_openpi_array(actions) -> np.ndarray:
    """Stack Cosmos action list to (H, 7) float64 for vla4desk Coordinator."""
    if isinstance(actions, np.ndarray):
        arr = np.asarray(actions, dtype=np.float64)
    else:
        arr = np.stack([np.asarray(row, dtype=np.float64) for row in actions], axis=0)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"Expected actions with shape (H, 7), got {arr.shape}")
    return arr


def extract_predicted_future_proprio(result: dict) -> np.ndarray | None:
    """Extract predicted future proprio as a flat (9,) vector."""
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



@dataclass
class Vla4deskServeConfig:
    """Mirrors scripts/validate_vla4desk_finetuned_lora_inference.PolicyEvalConfig."""

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
    seed: int = 195


class Vla4deskOpenPIBridgePolicy(BasePolicy):
    """OpenPI WebSocket policy adapter around Cosmos Policy get_action."""

    def __init__(self, cfg: Vla4deskServeConfig):
        self._cfg = cfg
        self._dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
        init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
        logger.info("Loading Cosmos Policy checkpoint: %s", cfg.ckpt_path)
        self._model, _ = get_model(cfg)
        self._metadata = {
            "policy": "cosmos_vla4desk_franka_lora",
            "config": cfg.config,
            "ckpt_path": cfg.ckpt_path,
            "chunk_size": cfg.chunk_size,
            "suite": cfg.suite,
        }

    @property
    def metadata(self) -> dict:
        return self._metadata

    def infer(self, obs: dict) -> dict:
        cosmos_obs, task_description = openpi_obs_to_cosmos(obs)
        with torch.inference_mode():
            result = get_action(
                self._cfg,
                self._model,
                self._dataset_stats,
                cosmos_obs,
                task_description,
                seed=self._cfg.seed,
                randomize_seed=False,
                num_denoising_steps_action=self._cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=True,
            )
        actions = actions_to_openpi_array(result["actions"])
        future_proprio = extract_predicted_future_proprio(result)

        response = {
            "actions": actions,
            "proprio": result["proprio"].tolist(),
            "future_proprio": future_proprio.tolist() if future_proprio is not None else None,
            "value_prediction": result.get("value_prediction"),
        }

        future_images = result.get("future_image_predictions")
        if future_images:
            response["future_images"] = {
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in future_images.items()
                if v is not None
            }

        return response

    def reset(self) -> None:
        pass
