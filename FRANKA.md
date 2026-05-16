# VLA4Desk: Franka 真机数据训练

## 系统环境

- GPU: RTX 3090 (24GB VRAM)
- 主模型: Cosmos Policy Predict2-2B（推理 ~6-9GB）
- 文本编码器: T5-11B（仅预处理用，~22GB bf16）

## 安装

### 方式一：Docker（推荐）

```bash
# 构建镜像
cd /media/czl/sata/franka_my_code/cosmos-policy
docker build -t cosmos-policy docker

# 启动容器
docker run \
  -u root \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v $HOME/.cache:/home/cosmos/.cache \
  -v /media/czl/sata/franka_my_code:/workspace \
  --gpus all \
  --ipc=host \
  -it --rm \
  -w /workspace/cosmos-policy \
  --entrypoint bash \
  cosmos-policy

# 容器内安装依赖
uv sync --extra cu128 --python 3.10
```

### 方式二：直接在宿主机安装

```bash
cd /media/czl/sata/franka_my_code/cosmos-policy

# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建 venv 并安装
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[cu128]"
```

## 数据

真机采集数据路径: `/media/czl/sata/franka_my_code/vla4desk/collected`

包含 12 个 task，共 135 条唯一指令:
- `banana_on_bowl`, `banana_on_plate`
- `pear_on_bowl`, `pear_on_cup`, `pear_on_plate`
- `strawberry_on_bowl`, `strawberry_on_cup`, `strawberry_on_place`
- `stack_blocks`, `simple_pick_place`
- `fruits_in_contrainers`, `final_clean`

### 提取 prompts（可跳过，用于检查）

```bash
python scripts/extract_vla4desk_prompts.py
```

## T5 Embedding 预计算

T5-11B 用于将文本指令编码为 embeddings，仅需在**预处理阶段**运行一次。

```bash
# 在容器内或宿主机虚拟环境中运行
uv run -m cosmos_policy.datasets.save_vla4desk_t5_text_embeddings \
  --data_root /media/czl/sata/franka_my_code/vla4desk/collected \
  --output_dir users/user/data/vla4desk
```

输出: `users/user/data/vla4desk/t5_embeddings.pkl`

> **注意事项**: T5-11B 在 bf16 下约占用 22GB VRAM。RTX 3090 24GB 能跑但较极限。
> 如果 OOM，可在 CPU 上运行（添加环境变量 `CUDA_VISIBLE_DEVICES=""`），但速度会很慢。

## 训练

（待补充 — 需要根据具体数据格式配置 dataset 和训练脚本）

## 推理

（待补充 — 加载 checkpoint 进行真机推理部署）
