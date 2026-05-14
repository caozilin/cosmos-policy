# Cosmos Policy: 用于视觉运动控制与规划的微调视频模型

<p align="center">
  <a href="https://arxiv.org/abs/2601.16163">论文</a>&nbsp | <a href="https://research.nvidia.com/labs/dir/cosmos-policy/">项目主页</a>&nbsp | 🤗 <a href="https://huggingface.co/collections/nvidia/cosmos-policy">模型与训练数据</a>&nbsp | <a href="https://youtu.be/V2qdFD9n5BM">介绍视频</a>
</p>

## 系统要求

仅使用基础 Cosmos Policy 进行推理（即不使用基于模型的规划）：
* 1 块 GPU，6.8 GB VRAM（适用于 LIBERO 仿真基准任务）
* 1 块 GPU，8.9 GB VRAM（适用于 RoboCasa 仿真基准任务）
* 1 块 GPU，6.0 GB VRAM（适用于 ALOHA 机器人任务）

使用 Cosmos Policy + 基于模型的规划（best-of-N 搜索）进行 ALOHA 机器人任务推理：
* 最低配置（串行推理）：1 块 GPU，10.0 GB VRAM
* 推荐配置（并行推理）：N 块 GPU，每块 10.0 GB VRAM

训练：
* 一般建议至少使用 1 个节点，包含 8 块 80GB GPU。在 Cosmos Policy 论文的实验中，我们使用 8 块 80GB GPU（H100）训练 48 小时进行小规模 ALOHA 机器人数据微调（<200 个演示），使用 32 块 80GB GPU（H100）训练 48 小时进行 RoboCasa 训练（1200 个演示），使用 64 块 80GB GPU（H100）训练 48 小时进行 LIBERO 训练（2000 个演示）。如果 GPU 数量较少，可以使用梯度累积来增加总批大小，我们发现这比使用更小批大小进行更多梯度步数收敛更快。

## 快速开始

首先，按照 [SETUP.md](SETUP.md) 中的说明设置 Docker 容器。

然后，在 Docker 容器中通过以下命令进入 Python 环境：`uv run --extra cu128 --group libero --python 3.10 python`。

接下来，运行以下 Python 代码来生成：(1) 机器人动作，(2) 预测的未来状态（由机器人本体感受和未来图像观测表示），和 (3) 未来状态价值（预期累积奖励）：

```python
import pickle
import torch
from PIL import Image
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    load_dataset_stats,
    init_t5_text_embeddings_cache,
    get_t5_embedding_from_cache,
)

# 实例化配置（关于配置的定义参见 cosmos_policy/experiments/robot/libero/run_libero_eval.py 中的 PolicyEvalConfig）
cfg = PolicyEvalConfig(
    config="cosmos_predict2_2b_480p_libero__inference_only",
    ckpt_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
    config_file="cosmos_policy/config/config.py",
    dataset_stats_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
    t5_text_embeddings_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
    use_wrist_image=True,
    use_proprio=True,
    normalize_proprio=True,
    unnormalize_actions=True,
    chunk_size=16,
    num_open_loop_steps=16,
    trained_with_image_aug=True,
    use_jpeg_compression=True,
    flip_images=True,  # 仅适用于 LIBERO；图像渲染是倒置的
    num_denoising_steps_action=5,
    num_denoising_steps_future_state=1,
    num_denoising_steps_value=1,
)
# 加载数据集统计信息用于动作/本体感受归一化
dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
# 初始化 T5 文本嵌入缓存
init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
# 加载模型
model, cosmos_config = get_model(cfg)
# 加载样本观测：
#   observation (dict): {
#     "primary_image": 主要第三人称图像,
#     "wrist_image": 手腕安装摄像头图像,
#     "proprio": 机器人本体感受状态,
#   }
with open("cosmos_policy/experiments/robot/libero/sample_libero_10_observation.pkl", "rb") as file:
    observation = pickle.load(file)
    task_description = "put both the alphabet soup and the tomato sauce in the basket"
# 生成机器人动作、未来状态（本体感受 + 图像）和价值
action_return_dict = get_action(
    cfg,
    model,
    dataset_stats,
    observation,
    task_description,
    num_denoising_steps_action=cfg.num_denoising_steps_action,
    generate_future_state_and_value_in_parallel=True,
)
# 打印动作
print(f"Generated action chunk: {action_return_dict['actions']}")
# 保存未来图像预测（第三人称图像和手腕图像）
img_path1, img_path2 = "future_image.png", "future_wrist_image.png"
Image.fromarray(action_return_dict['future_image_predictions']['future_image']).save(img_path1)
Image.fromarray(action_return_dict['future_image_predictions']['future_wrist_image']).save(img_path2)
print(f"Saved future image predictions to:\n\t{img_path1}\n\t{img_path2}")
# 打印价值
print(f"Generated value: {action_return_dict['value_prediction']}")
```

如果遇到运行时错误，可能需要先通过 `uv run --extra cu128 --group libero --python 3.10 python` 进入 Python 环境，再运行上述代码。

## 安装

关于环境设置说明，请参见 [SETUP.md](SETUP.md)。

## 训练与评估

关于在 LIBERO 仿真基准任务套件上进行微调/评估，请参见 [LIBERO.md](LIBERO.md)。

关于在 RoboCasa 仿真基准任务上进行微调/评估，请参见 [ROBOCASA.md](ROBOCASA.md)。

关于在真实世界 ALOHA 机器人任务上进行微调/评估，请参见 [ALOHA.md](ALOHA.md)。

## 支持

如果遇到任何问题，请提交新的 GitHub Issue。对于关键的阻塞性问题，请发送邮件至 Moo Jin Kim（moojink@cs.stanford.edu）以引起他的注意。

## 引用

如果您的工作中使用了我们的代码，请引用[我们的论文](https://arxiv.org/abs/2601.16163)：

```bibtex
@article{kim2026cosmos,
  title={Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning},
  author={Kim, Moo Jin and Gao, Yihuai and Lin, Tsung-Yi and Lin, Yen-Chen and Ge, Yunhao and Lam, Grace and Liang, Percy and Song, Shuran and Liu, Ming-Yu and Finn, Chelsea and Gu, Jinwei},
  journal={arXiv preprint arXiv:2601.16163},
  year={2026}
}
```
