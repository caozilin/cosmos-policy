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
run_libero_eval_with_json.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
Records actions and value predictions per timestep to JSON files.

Usage:
    export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN to your Hugging Face access token}"
    HF_HOME=/media/czl/sata/franka_my_code/cosmos-policy/hf_cache \
    uv run --extra cu128 --group libero --python 3.10 \
        python -m cosmos_policy.experiments.robot.libero.run_libero_eval_with_json \
        --config cosmos_predict2_2b_480p_libero__inference_only \
        --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
        --config_file cosmos_policy/config/config.py \
        --use_wrist_image True \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 16 \
        --num_open_loop_steps 16 \
        --task_suite_name libero_10 \
        --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
        --json_output_dir cosmos_policy/experiments/robot/libero/json_logs/ \
        --seed 195 \
        --deterministic True \
        --run_id_note eval_with_json \
        --num_trials_per_task 1
"""

import json
import logging
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import draccus
import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
import tqdm
from libero.libero import benchmark

from cosmos_policy.experiments.robot.cosmos_utils import (
    WorkerPoolManager,
    get_action,
    get_future_state_prediction,
    get_model,
    get_planning_model,
    get_qvalue_prediction,
    get_value_prediction,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    query_model_parallel,
)
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    save_rollout_video,
    save_rollout_video_with_future_image_predictions,
)
from cosmos_policy.experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    log_message,
    setup_logging,
)
from cosmos_policy.utils.utils import jpeg_encode_image, set_seed_everywhere

CURR_STATE_START_LATENT_IDX, CURR_STATE_END_LATENT_IDX = 1, 3
FUTURE_STATE_START_LATENT_IDX, FUTURE_STATE_END_LATENT_IDX = 5, 7


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}


@dataclass
class PolicyEvalConfig:
    suite: str = "libero"

    model_family: str = "cosmos"
    config: str = ""
    ckpt_path: str = ""
    planning_model_config_name: str = ""
    planning_model_ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"

    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    flip_images: bool = True
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
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""
    trained_with_image_aug: bool = True
    chunk_size: int = 16
    num_open_loop_steps: int = 16

    deterministic: bool = True
    deterministic_reset: bool = False
    deterministic_reset_seed: int = None

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
    use_parallel_inference: bool = False
    available_gpus: str = "0,1,2,3,4,5,6,7"
    parallel_timeout: int = 15

    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    local_log_dir: str = "./experiments/logs"
    json_output_dir: str = "./experiments/json_logs"
    run_id_note: Optional[str] = None

    use_wandb: bool = False
    wandb_entity: str = "YOUR_ENTITY"
    wandb_project: str = "YOUR_PROJECT"

    seed: int = 7
    randomize_seed: bool = False

    data_collection: bool = False
    jpeg_compress: bool = True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def validate_config(cfg: PolicyEvalConfig) -> None:
    assert cfg.ckpt_path is not None, "ckpt_path must not be None!"
    if "image_aug" in str(cfg.ckpt_path):
        assert cfg.trained_with_image_aug, (
            "Expecting `trained_with_image_aug==True` because model was trained with image augmentations!"
        )
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def check_unnorm_key(cfg: PolicyEvalConfig, model) -> None:
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in Cosmos Policy `norm_stats`!"
    cfg.unnorm_key = unnorm_key


def load_initial_states(cfg: PolicyEvalConfig, task_suite, task_id: int, log_file=None):
    initial_states = task_suite.get_task_init_states(task_id)
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size, flip_images: bool = False):
    img = get_libero_image(obs, flip_images)
    wrist_img = get_libero_wrist_image(obs, flip_images)
    observation = {
        "primary_image": img,
        "wrist_image": wrist_img,
        "proprio": np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"])),
    }
    return observation


def run_episode(
    cfg: PolicyEvalConfig,
    env,
    task_description: str,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    resize_size,
    initial_state=None,
    log_file=None,
):
    if cfg.deterministic_reset:
        reset_seed = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
        set_seed_everywhere(reset_seed)
    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    if cfg.num_open_loop_steps != cfg.chunk_size:
        print(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match cfg.chunk_size "
            f"{cfg.chunk_size}! For best performance (in terms of both speed and success rate), we "
            "recommend executing the full action chunk."
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    replay_images = []
    replay_wrist_images = [] if cfg.use_wrist_image else None
    future_image_predictions_list = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    base_seed = cfg.seed

    if cfg.data_collection:
        primary_images_list = []
        wrist_images_list = []
        proprio_list = []
        actions_list = []

    success = False

    json_data = {
        "task_description": task_description,
        "task_suite": cfg.task_suite_name,
        "seed": cfg.seed,
        "steps": [],
    }
    current_inference_value = None

    try:
        NUM_STEPS_WAIT = 10
        while t < max_steps + NUM_STEPS_WAIT:
            if os.environ.get("DETERMINISTIC", "").lower() == "true":
                seed = 0
                set_seed_everywhere(seed)

            if t < NUM_STEPS_WAIT:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation = prepare_observation(obs, resize_size, cfg.flip_images)
            replay_images.append(observation["primary_image"])
            if replay_wrist_images is not None:
                replay_wrist_images.append(observation["wrist_image"])

            if cfg.data_collection:
                primary_images_list.append(observation["primary_image"])
                wrist_images_list.append(observation["wrist_image"])
                proprio_list.append(observation["proprio"])

            if len(action_queue) == 0:
                best_actions = None
                best_future_predictions = None

                num_queries = cfg.num_queries_best_of_n

                if cfg.use_parallel_inference and num_queries > 1 and worker_pool and worker_pool.initialized:
                    start_time = time.time()
                    query_results = query_model_parallel(
                        cfg, observation, task_description, worker_pool, cfg.parallel_timeout
                    )
                    total_query_time = time.time() - start_time
                    log_message(
                        f"Parallel queries completed: {len(query_results)} results in {total_query_time:.3f}s", log_file
                    )
                else:
                    query_results = []
                    for query_idx in range(num_queries):
                        actions_by_depth = []
                        future_image_predictions_by_depth = []
                        value_predictions_by_depth = []
                        return_dict = {}
                        start_time = time.time()
                        action_return_dict = get_action(
                            cfg,
                            model,
                            dataset_stats,
                            observation,
                            task_description,
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_action=cfg.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=not (
                                cfg.ar_future_prediction or cfg.ar_value_prediction or cfg.ar_qvalue_prediction
                            ),
                        )
                        query_time = time.time() - start_time
                        log_message(
                            f"Query {query_idx + 1}/{num_queries}: Action query time = {query_time:.3f} sec", log_file
                        )
                        return_dict["actions"] = action_return_dict["actions"]
                        actions_by_depth.append(return_dict["actions"])

                        if cfg.ar_future_prediction:
                            start_time = time.time()
                            future_state_return_dict = get_future_state_prediction(
                                cfg,
                                model=planning_model if planning_model is not None else model,
                                data_batch=action_return_dict["data_batch"],
                                generated_latent_with_action=action_return_dict["generated_latent"],
                                orig_clean_latent_frames=action_return_dict["orig_clean_latent_frames"],
                                future_proprio_latent_idx=action_return_dict["latent_indices"]["future_proprio_latent_idx"],
                                future_wrist_image_latent_idx=action_return_dict["latent_indices"]["future_wrist_image_latent_idx"],
                                future_wrist_image2_latent_idx=action_return_dict["latent_indices"]["future_wrist_image2_latent_idx"],
                                future_image_latent_idx=action_return_dict["latent_indices"]["future_image_latent_idx"],
                                future_image2_latent_idx=action_return_dict["latent_indices"]["future_image2_latent_idx"],
                                seed=cfg.seed + query_idx,
                                randomize_seed=cfg.randomize_seed,
                                num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                                use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                                num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                                future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
                            )
                            query_time = time.time() - start_time
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Future state prediction query time = {query_time:.3f} sec",
                                log_file,
                            )
                            return_dict["future_image_predictions"] = future_state_return_dict["future_image_predictions"]
                            future_image_predictions_by_depth.append(return_dict["future_image_predictions"])
                        else:
                            return_dict["future_image_predictions"] = action_return_dict["future_image_predictions"]

                        if cfg.ar_value_prediction:
                            start_time = time.time()
                            value_return_dict = get_value_prediction(
                                cfg,
                                model=planning_model if planning_model is not None else model,
                                data_batch=action_return_dict["data_batch"],
                                future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                                seed=cfg.seed + query_idx,
                                randomize_seed=cfg.randomize_seed,
                                num_denoising_steps_value=cfg.num_denoising_steps_value,
                                use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                                num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                            )
                            query_time = time.time() - start_time
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Value prediction query time = {query_time:.3f} sec",
                                log_file,
                            )
                            return_dict["value_prediction"] = value_return_dict["value_prediction"]
                            value_predictions_by_depth.append(return_dict["value_prediction"])
                        elif cfg.ar_qvalue_prediction:
                            start_time = time.time()
                            value_return_dict = get_qvalue_prediction(
                                cfg,
                                model=planning_model if planning_model is not None else model,
                                data_batch=action_return_dict["data_batch"],
                                action_sample=action_return_dict["generated_latent"],
                                seed=cfg.seed + query_idx,
                                randomize_seed=cfg.randomize_seed,
                                num_denoising_steps_value=cfg.num_denoising_steps_value,
                                use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                                num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                            )
                            query_time = time.time() - start_time
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Value prediction query time = {query_time:.3f} sec",
                                log_file,
                            )
                            return_dict["value_prediction"] = value_return_dict["value_prediction"]
                            value_predictions_by_depth.append(return_dict["value_prediction"])
                        else:
                            return_dict["value_prediction"] = action_return_dict["value_prediction"]
                            value_predictions_by_depth.append(return_dict["value_prediction"])

                        return_dict["future_image_predictions_by_depth"] = future_image_predictions_by_depth
                        return_dict["value_predictions_by_depth"] = value_predictions_by_depth
                        return_dict["actions_by_depth"] = actions_by_depth
                        query_results.append(return_dict)

                log_message(f"t={t}: Current base seed: {base_seed}", log_file)
                for query_idx, return_dict in enumerate(query_results):
                    predicted_value = return_dict["value_prediction"]
                    log_message(
                        f"Query {query_idx + 1}/{num_queries} (seed {cfg.seed + query_idx}): Predicted value = {predicted_value:.4f}",
                        log_file,
                    )
                seed_to_return_dict = {
                    cfg.seed + query_idx: (
                        return_dict["actions"],
                        return_dict["future_image_predictions"],
                        return_dict["value_prediction"],
                    )
                    for query_idx, return_dict in enumerate(query_results)
                }
                best_seed, best_return_dict = max(seed_to_return_dict.items(), key=lambda x: x[1][2])
                best_actions = best_return_dict[0]
                best_future_predictions = best_return_dict[1]
                best_value_predictions = best_return_dict[2]

                current_inference_value = float(best_value_predictions)

                action_queue.extend(best_actions)
                future_image_predictions_list.append(best_future_predictions)
                log_message(f"t={t}: Selected seed {best_seed} with value = {best_value_predictions:.4f}", log_file)

            action = action_queue.popleft()

            json_data["steps"].append({
                "t": t,
                "action": action.tolist(),
                "value_prediction": current_inference_value,
            })

            if cfg.data_collection:
                actions_list.append(action.copy())

            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        error_msg = f"Episode error: {e}"
        traceback_str = traceback.format_exc()
        log_message(f"{error_msg}\nFull traceback:\n{traceback_str}", log_file)

    json_data["success"] = success
    json_data["total_steps"] = t

    if cfg.data_collection:
        collected_data = dict(
            primary_images=np.stack(primary_images_list, axis=0),
            wrist_images=np.stack(wrist_images_list, axis=0),
            proprio=np.stack(proprio_list, axis=0),
            actions=np.stack(actions_list, axis=0),
            success=success,
        )
        if len(future_image_predictions_list) > 0:
            if cfg.use_third_person_image:
                future_primary_images = [
                    x["future_image"] for x in future_image_predictions_list if x["future_image"] is not None
                ]
                if len(future_primary_images) > 0:
                    collected_data["future_primary_images"] = np.stack(future_primary_images, axis=0)
            if (
                cfg.use_wrist_image
                and "future_wrist_image" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_wrist_image"] is not None
            ):
                future_wrist_images = [x["future_wrist_image"] for x in future_image_predictions_list]
                collected_data["future_wrist_images"] = np.stack(future_wrist_images, axis=0)
    else:
        collected_data = None

    return success, replay_images, replay_wrist_images, future_image_predictions_list, collected_data, json_data


def run_task(
    cfg: PolicyEvalConfig,
    task_suite,
    task_id: int,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    resize_size,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        success, replay_images, replay_wrist_images, future_image_predictions_list, collected_data, json_data = run_episode(
            cfg,
            env,
            task_description,
            model,
            planning_model,
            dataset_stats,
            worker_pool,
            resize_size,
            initial_state,
            log_file,
        )

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        os.makedirs(cfg.json_output_dir, exist_ok=True)
        json_filename = f"task={task_id}_episode={total_episodes}_success={success}_{DATE_TIME}.json"
        json_filepath = os.path.join(cfg.json_output_dir, json_filename)
        with open(json_filepath, "w") as f:
            json.dump(json_data, f, indent=2)
        log_message(f"Saved JSON to {json_filepath}", log_file)

        save_rollout_video(
            replay_images,
            total_episodes,
            success=success,
            task_description=task_description,
            log_file=log_file,
        )

        future_primary_image_predictions = None
        if cfg.use_third_person_image:
            future_primary_image_predictions = [x["future_image"] for x in future_image_predictions_list]
        future_wrist_image_predictions = None
        if cfg.use_wrist_image:
            future_wrist_image_predictions = [x["future_wrist_image"] for x in future_image_predictions_list]
        save_rollout_video_with_future_image_predictions(
            replay_images,
            total_episodes,
            success=success,
            task_description=task_description,
            chunk_size=cfg.chunk_size,
            num_open_loop_steps=cfg.num_open_loop_steps,
            rollout_wrist_images=replay_wrist_images,
            future_primary_image_predictions=future_primary_image_predictions,
            future_wrist_image_predictions=future_wrist_image_predictions,
            log_file=log_file,
            show_diff=False,
        )

        if cfg.data_collection and collected_data is not None:
            def _save_episode_data():
                ep_filename = f"episode_data--suite={cfg.task_suite_name}--{DATE_TIME}--task={task_id}--ep={total_episodes}--success={success}--{cfg.run_id_note}.hdf5"
                rollout_data_dir = os.path.join(cfg.local_log_dir, "rollout_data")
                os.makedirs(rollout_data_dir, exist_ok=True)
                ep_filepath = os.path.join(rollout_data_dir, ep_filename)
                with h5py.File(ep_filepath, "w") as f:
                    for k, v in collected_data.items():
                        if isinstance(v, np.ndarray):
                            is_image = v.ndim == 4 and v.shape[-1] == 3 and v.dtype == np.uint8
                            if is_image and cfg.jpeg_compress:
                                jpeg_list = [jpeg_encode_image(frame, quality=95) for frame in v]
                                dt = h5py.vlen_dtype(np.dtype("uint8"))
                                f.create_dataset(k + "_jpeg", data=jpeg_list, dtype=dt)
                            else:
                                f.create_dataset(k, data=v)
                        else:
                            f.attrs[k] = v
                    f.attrs["task_description"] = task_description

            _save_episode_data()

        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{cfg.task_suite_name}/{task_description}": task_success_rate,
                f"num_episodes/{cfg.task_suite_name}/{task_description}": task_episodes,
                f"num_successes/{cfg.task_suite_name}/{task_description}": task_successes,
            },
        )

    return (
        total_episodes,
        total_successes,
    )


@draccus.wrap()
def eval_libero_with_json(cfg: PolicyEvalConfig) -> float:
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"

    if cfg.use_parallel_inference:
        mp.set_start_method("spawn", force=True)

    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    worker_pool = None
    if cfg.use_parallel_inference:
        available_gpus = [int(gpu.strip()) for gpu in cfg.available_gpus.split(",")]
        available_gpus = available_gpus[: cfg.num_queries_best_of_n]
        worker_pool = WorkerPoolManager(cfg, dataset_stats, available_gpus)
        model = None
        planning_model = None
    else:
        model, cosmos_config = get_model(cfg)
        assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
            f"Mismatch found between train and test chunk sizes! Train: {cosmos_config.dataloader_train.dataset.chunk_size}, Test: {cfg.chunk_size}"
        )
        worker_pool = None
        if cfg.planning_model_ckpt_path != "":
            planning_model, _ = get_planning_model(cfg)
        else:
            planning_model = None

    resize_size = get_image_resize_size(cfg.model_family)

    log_file, local_log_filepath, run_id = setup_logging(
        cfg=cfg,
        task_identifier=cfg.task_suite_name,
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
    )
    log_message(f"Eval config: {cfg}", log_file)

    if cfg.use_parallel_inference and worker_pool:
        log_message(f"Parallel inference enabled on GPUs: {available_gpus}", log_file)
        log_message(f"Parallel timeout: {cfg.parallel_timeout}s", log_file)
        log_message(f"Multiprocessing start method: {mp.get_start_method()}", log_file)
        for gpu_id in available_gpus:
            if gpu_id >= torch.cuda.device_count():
                log_message(
                    f"Warning: GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs found)", log_file
                )
        try:
            log_message("Starting worker pool...", log_file)
            worker_pool.start_workers()
            log_message("Worker pool started successfully", log_file)
        except Exception as e:
            error_msg = f"Failed to start worker pool: {e}"
            traceback_str = traceback.format_exc()
            log_message(f"{error_msg}\nFull traceback:\n{traceback_str}", log_file)
            log_message("Disabling parallel inference for this run", log_file)
            worker_pool = None
    else:
        log_message("Using serial inference (parallel inference disabled)", log_file)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Number of tasks: {num_tasks}", log_file)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        (
            total_episodes,
            total_successes,
        ) = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            planning_model,
            dataset_stats,
            worker_pool,
            resize_size,
            total_episodes,
            total_successes,
            log_file,
        )

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{cfg.task_suite_name}/total": final_success_rate,
                f"num_episodes/{cfg.task_suite_name}/total": total_episodes,
                f"num_successes/{cfg.task_suite_name}/total": total_successes,
            },
        )
        wandb.save(local_log_filepath)

    if worker_pool:
        try:
            worker_pool.shutdown()
        except Exception as e:
            error_msg = f"Error shutting down worker pool: {e}"
            traceback_str = traceback.format_exc()
            log_message(f"{error_msg}\nFull traceback:\n{traceback_str}", log_file)

    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero_with_json()
