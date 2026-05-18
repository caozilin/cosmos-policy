# VLA4Desk: Franka 真机数据训练

与上游环境说明一致，请以 [SETUP.md](SETUP.md)、[README.md](README.md) 为准。本文只补充 Franka / VLA4Desk 相关路径与命令。

## 系统环境

- GPU: RTX 3090 (24GB VRAM) 或云端 4090 等；16G 卡需 T5 CPU/offload
- 主模型: Cosmos Policy Predict2-2B（推理 ~6-9GB）
- 文本编码器: T5-11B **Encoder**（仅预处理；默认 **`fp32`** 约 **20–22GB** 显存；`bf16` 约 **10–12GB**）

## 安装

**不要**使用 `pip install -e ".[cu128]"`、`python3 -m venv`（系统默认可能是 3.13）、也不要把本机 `.venv` rsync 到另一台机器。应使用 **`uv sync --extra cu128 --python 3.10`**（与 [LIBERO.md](LIBERO.md) 相同，但 Franka 工作**不需要** `--group libero`）。

### 方式一：Docker（推荐，与 SETUP.md 一致）

构建与启动见 [SETUP.md](SETUP.md)。进入容器后，在项目根目录执行：

```bash
uv sync --extra cu128 --python 3.10
source .venv/bin/activate
```

若 `vla4desk` 数据在仓库外，启动容器时额外挂载父目录（示例）：

```bash
docker run \
  -u root \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -v $HOME/.cache:/home/cosmos/.cache \
  -v $(pwd):/workspace \
  -v /path/to/parent-of-vla4desk:/data \
  --gpus all \
  --ipc=host \
  -it --rm \
  -w /workspace \
  --entrypoint bash \
  cosmos-policy
```

### 方式二：宿主机（无 Docker）

```bash
cd /path/to/cosmos-policy

# 若使用 conda，先退出，避免抢用 base 的 python
conda deactivate

# 安装 uv（若未安装）：https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

# 若此前用错误方式建过 venv（如 3.13 / pip），先删除再 sync
rm -rf .venv

uv sync --extra cu128 --python 3.10
source .venv/bin/activate
which python && python -V   # 应为 .venv/bin/python 且 3.10.x
```

需要 LIBERO 仿真评测时，才使用 `uv sync --extra cu128 --group libero --python 3.10`（见 [LIBERO.md](LIBERO.md)）。

## 数据

真机采集默认路径: `vla4desk/collected/`（相对仓库根；大文件不进 git）

包含 12 个 task，共 135 条唯一指令:
- `banana_on_bowl`, `banana_on_plate`
- `pear_on_bowl`, `pear_on_cup`, `pear_on_plate`
- `strawberry_on_bowl`, `strawberry_on_cup`, `strawberry_on_place`
- `stack_blocks`, `simple_pick_place`
- `fruits_in_contrainers`, `final_clean`

### 提取唯一指令（prompt）

`scripts/extract_vla4desk_prompts.py` 会遍历采集目录下所有 `epo_*/data.json`，汇总去重后的 `prompt`（及对应 `task_name`）。用于转换前核对语言指令、统计条数，或单独生成 `prompts.txt`。

默认扫描 `<repo>/vla4desk/collected`；其它路径用 `--data-root`。

```bash
source .venv/bin/activate
cd /path/to/cosmos-policy

# 列出全部唯一 prompt（带 task 名）
python scripts/extract_vla4desk_prompts.py

# 只打印 prompt 文本，一行一条
python scripts/extract_vla4desk_prompts.py --quiet

# 写入文件（每行一条，可供 precompute_t5 或人工检查）
python scripts/extract_vla4desk_prompts.py -o ./vla4desk/prompts.txt
```

说明：

- 转换脚本 `convert_vla4desk_to_libero_hdf5.py` 也会在输出目录生成 `prompts.txt`；本工具适合**转换前**从原始采集数据扫一遍，或与转换结果交叉核对。
- 同一 `prompt` 在多个 episode 里重复出现时只保留一条；`task_name` 取首次出现的 episode 所在任务目录名。

### 从 eval `telemetry.jsonl` 提取 / 合并 prompt

`scripts/extract_eval_prompts.py` 递归扫描 eval 目录下所有 `telemetry.jsonl`（如 `.../eval/unseen/abstract_ins/epo_10/telemetry.jsonl`），读取每行 JSON 的 `prompt` 字段并去重。

将 eval 的 `telemetry.jsonl` 树放到 `vla4desk/eval/`（或任意目录并用 `--eval-root` 指定）：

```bash
# 仅 eval → vla4desk/prompts.txt（默认 --eval-root ./vla4desk/eval）
python scripts/extract_eval_prompts.py -o ./vla4desk/prompts.txt

# 采集 135 条 + eval 新增，合并去重
python scripts/extract_eval_prompts.py --merge-committed-prompts -o ./vla4desk/prompts.txt
```

`--merge-committed-prompts` 会先并入 git 中原来的 `vla4desk/prompts.txt`（采集数据 135 条），再追加 eval 里尚未出现的 prompt。也可用 `--merge-from other.txt` 指定其它列表。

## 转换为 LIBERO 格式 HDF5

采集目录为 `vla4desk/collected/<task>/epo_*/`（`cam1.mp4`、`cam2.mp4`、`data.json`）。用 `scripts/convert_vla4desk_to_libero_hdf5.py` 转为 Cosmos Policy `LIBERODataset` 可读的演示 HDF5（字段与 [LIBERO.md](LIBERO.md) 一致）。

**约定（与 OpenPI `convert_vla4desk_data_to_lerobot.py` 一致）：**

- `action`：按 `data.json` **原样**写入，不除以 `action_scale`
- 帧：与 JSON / 视频 **逐帧对齐**，不删静止帧
- 语言：`prompt` 写入文件名（`put_the_banana_in_the_red_bowl_demo.hdf5`）及 HDF5 属性 `task_description`；`LIBERODataset` 仍从文件名解析 `command`

**输出目录示例：**

```text
datasets/VLA4Desk-Franka/success_only/vla4desk_franka/
  put_the_banana_in_the_red_bowl_demo.hdf5
  prompts.txt
  t5_embeddings.pkl          # 若源 pkl 里已有对应 prompt 则自动写入子集
  conversion_manifest.json
```

**转换命令：**

```bash
source .venv/bin/activate
cd /path/to/cosmos-policy

python scripts/convert_vla4desk_to_libero_hdf5.py \
  --suite-name vla4desk_franka
```

（`--input` / `--output` 默认分别为 `vla4desk/collected` 与 `datasets/VLA4Desk-Franka/success_only`。）

```bash
# 显式指定路径时：
python scripts/convert_vla4desk_to_libero_hdf5.py \
  --input ./vla4desk/collected \
  --output ./datasets/VLA4Desk-Franka/success_only \
  --suite-name vla4desk_franka
```

可选参数：

```bash
# 与 LIBERO regen 一致：256 分辨率 + JPEG 压缩（省磁盘）
python scripts/convert_vla4desk_to_libero_hdf5.py ... --resize 256 --jpeg-compress

# 保留原始 640×480
python scripts/convert_vla4desk_to_libero_hdf5.py ... --resize 0

# 指定 T5 源（默认会尝试 vla4desk/t5_embeddings.pkl）
python scripts/convert_vla4desk_to_libero_hdf5.py ... \
  --t5-embeddings ./vla4desk/t5_embeddings.pkl
```

训练时 `data_dir` 指向 `.../success_only`，`t5_text_embeddings_path` 指向 `.../vla4desk_franka/t5_embeddings.pkl`（若该文件不存在，见下方 T5 一节先对 `prompts.txt` 预计算）。

> **耗时说明**：转换**不会**加载 T5-11B 模型；慢在逐条读 MP4、默认 `resize 256`（PIL 逐帧）、以及 HDF5 的 `gzip` 压图像（135 条约需十几分钟，终端会显示 `Episodes` 进度条）。若只是复制已有 embedding，末尾 `pickle.load` 大 pkl 也可能卡几十秒。加速可加 `--jpeg-compress`，或 `--skip-t5` 跳过 pkl 复制。

## T5 Embedding 预计算

须先完成上方 **安装**（`uv sync`）。输入为每行一条指令的 `prompts.txt`（例如转换后目录下的 `vla4desk_franka/prompts.txt`，或仓库内 `vla4desk/prompts.txt`），输出为同目录或指定路径的 `t5_embeddings.pkl`。

若转换脚本已根据已有 pkl 写出**子集** `t5_embeddings.pkl`，但仍有 prompt 缺 embedding，对完整 `prompts.txt` 再跑一遍即可覆盖/补全：

```bash
python scripts/precompute_t5_embeddings.py \
  -i ./datasets/VLA4Desk-Franka/success_only/vla4desk_franka/prompts.txt \
  -o ./datasets/VLA4Desk-Franka/success_only/vla4desk_franka/t5_embeddings.pkl
```

T5 权重目录：`hf_cache/models--google-t5--t5-11b/snapshots/<hash>/`（需含 `spiece.model`、`tokenizer.json`、`pytorch_model.bin`）。

**必须**设 **`HF_HUB_CACHE`** 指向仓库内 `hf_cache` 根目录（不要只设 `HF_HOME`）。在仓库根目录下可用 `export HF_HUB_CACHE=./hf_cache`。缓存齐全时设 `HF_HUB_OFFLINE=1`，避免误从网络下载 `spiece.model`。

**常见问题：**

- `TypeError: not a string`（加载 tokenizer）：未设 `HF_HUB_CACHE`，或缓存不完整。
- 进度条 `0/135` 很久不动、GPU 空：正在从磁盘读 ~43G 权重到 CPU；第一次完成后会变为 `1/135`。
- `snapshots/` 下有空目录（无 `spiece.model`）：删除空 snapshot，只保留完整 hash 目录。

```bash
source .venv/bin/activate
export HF_HUB_CACHE=/path/to/cosmos-policy/hf_cache
export HF_HUB_OFFLINE=1

# 推荐：24G+ / 4090，整 Encoder 上 GPU（默认 fp32）
python scripts/precompute_t5_embeddings.py \
  --gpu 0 --device cuda \
  --hf-hub-cache "$HF_HUB_CACHE" --local-files-only \
  -i ./vla4desk/prompts.txt \
  -o ./datasets/VLA4Desk-Franka/success_only/vla4desk_franka/t5_embeddings.pkl

# 省显存：bf16 加载（输出 pkl 仍为 bf16）
python scripts/precompute_t5_embeddings.py --gpu 0 --device cuda --dtype bf16

# 16G 显存：GPU 限 12G + CPU offload（建议加 --dtype bf16）
python scripts/precompute_t5_embeddings.py --gpu 0 --device auto --max_gpu_mem_gib 12 --dtype bf16

# 纯 CPU（不占显存，较慢）
python scripts/precompute_t5_embeddings.py --gpu '' --device cpu
```

脚本参数：`--hf-hub-cache`、`--local-files-only`、`--dtype {fp32,bf16}`（默认 `fp32`）。

也可用 `uv run`（无需手动 activate）：

```bash
export HF_HUB_CACHE=/path/to/cosmos-policy/hf_cache
export HF_HUB_OFFLINE=1
uv run --extra cu128 --python 3.10 python scripts/precompute_t5_embeddings.py \
  --gpu 0 --device cuda --hf-hub-cache "$HF_HUB_CACHE" --local-files-only \
  -i ./vla4desk/prompts.txt \
  -o ./datasets/VLA4Desk-Franka/success_only/vla4desk_franka/t5_embeddings.pkl
```

> 预计算使用 `T5EncoderModel`（仅 Encoder）。默认 `fp32` 显存约 **20–22GB**；`--dtype bf16` 约 **10–12GB**。磁盘 `pytorch_model.bin` 为整包 checkpoint（~43G），读入时 **CPU 内存** 峰值仍可能暂时很高。

## 离线验证（可选，非训练必须）

`scripts/validate_vla4desk_libero_inference.py`：从转换后的 HDF5 读图 / proprio / action，用 **LIBERO 预训练权重** 做若干时刻的推理。语言条件使用 HDF5 的 **`task_description`（正常提示词）**，T5 来自 **`vla4desk_franka/t5_embeddings.pkl`**（须先对 `prompts.txt` 预计算）。

须安装 LIBERO 依赖（`--group libero`）。`export HF_HOME=...` 仅用于加载 LIBERO checkpoint / `libero_dataset_statistics.json`。

```bash
export HF_HOME=/path/to/cosmos-policy/hf_cache

uv run --extra cu128 --group libero --python 3.10 \
  python scripts/validate_vla4desk_libero_inference.py \
  --hdf5-path datasets/VLA4Desk-Franka/success_only/vla4desk_franka/put_the_yellow_cube_on_the_red_plate_demo.hdf5 \
  --t5-text-embeddings-path datasets/VLA4Desk-Franka/success_only/vla4desk_franka/t5_embeddings.pkl \
  --timestamps-json /path/to/vla4desk/collected/simple_pick_place/epo_1/data.json \
  --output-dir ./validation_outputs/vla4desk_epo_1_hdf5
```

输出：`validation_*.json`、`states_actions_*.csv`、以及 `comparisons/` 下输入图与预测 future 图的对比 PNG（默认 0–18 s、步长 2 s）。

## 训练

配置见 `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py`。

| 实验名 | 说明 |
|--------|------|
| **`cosmos_predict2_2b_480p_vla4desk_franka_135_demos_lora`** | **推荐**：PEFT LoRA，冻结 LIBERO 主干，只训 adapter |
| `cosmos_predict2_2b_480p_vla4desk_franka_135_demos` | 全参数微调（显存大，8×80GB 更合适） |

二者共同点：

- **数据**：`LIBERODataset`，`data_dir` = `datasets/VLA4Desk-Franka/success_only/vla4desk_franka/`（135 HDF5）
- **T5**：同目录 `t5_embeddings.pkl`
- **归一化**：数据目录内 `dataset_statistics.json`（**勿用** `libero_dataset_statistics.json`）
- **初始化**：`nvidia/Cosmos-Policy-LIBERO-Predict2-2B`（加载时把权重灌进 LoRA 的 `base_layer`，adapter 随机初始化）

LoRA 默认：`rank=32`，`alpha=32`，目标模块 `q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2`（与 `Text2WorldModelConfig` 一致）。依赖 `peft`（已在 `pyproject.toml`）。

### 路径与离线（可选）

仓库已默认：`BASE_DATASETS_DIR` / `HF_HOME` / checkpoint 输出均指向项目内路径（见 `cosmos_policy/config/output_paths.py`）。一般 **无需** 再 `export BASE_DATASETS_DIR="$(pwd)"`。

离线训练（权重已在 `hf_cache/`）：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

首次冷启动日志中应出现 `Loading consolidated pretrained weights from: ...Cosmos-Policy-LIBERO-Predict2-2B.pt`，不应是 `Training from scratch.`。

### 8×3090 LoRA 训练（推荐配方）

目标：**有效 batch ≈ 600**（非 LIBERO 论文的 1920），**每 1000 step 存盘**，**最多 30000 step**。W&B 项目名 **`cosmos-policy`**。

**勿用 `model=policy_ddp` 搭配 `..._lora` 实验**：Hydra 会用 `policy_ddp` 整段替换 experiment 里的 `model`，得到默认 `use_lora=False`，等于全参数 DDP 微调（显存与 checkpoint 含义都不同）。只切 DDP 时请用 `trainer.distributed_parallelism=ddp` + `model.config.fsdp_shard_size=1`。

\[
B_{\text{eff}} = \text{batch\_per\_GPU} \times 8 \times \text{grad\_accum} = 6 \times 8 \times 12 = 576 \approx 600
\]

```bash
cd /path/to/cosmos-policy

uv run --extra cu128 --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_vla4desk_franka_135_demos_lora" \
  trainer.distributed_parallelism=ddp \
  model.config.fsdp_shard_size=1 \
  ckpt_type=dcp \
  dataloader_train.batch_size=6 \
  trainer.grad_accum_iter=12 \
  trainer.max_iter=30000 \
  checkpoint.save_iter=1000 \
  trainer.logging_iter=10 \
  job.wandb_mode=online \
  job.project=cosmos-policy \
  job.group=cosmos_v2_finetune \
  job.name=cosmos_predict2_2b_480p_vla4desk_franka_135_demos_lora__8x3090_bs6_ga12 \
  optimizer.lr=1e-4
```

LoRA 实验默认继承 LIBERO 父项：`max_iter=1_000_000`、`save_iter=500`、`batch_size=4`、`grad_accum_iter=1`、并行 **FSDP**（`policy_fsdp`）。上表 CLI 覆盖 batch/步数/存盘；并行改为 DDP。仅写 `experiment=..._lora` 不写上述项时，**LoRA 仍在**，但是 FSDP 且会训很久。

**冷启动日志自检**（rank0）：应有 `Loading consolidated pretrained weights from: ...Cosmos-Policy-LIBERO-Predict2-2B.pt`、`Model uses LoRA, mapping checkpoint keys...`、`LoRA injection successful: ... trainable parameters out of ...`；不应是 `Training from scratch.`；可训练参数量应为全模型约 **0.x%** 量级。

| 项 | 本配方 |
|----|--------|
| GPU | 8×3090（24GB） |
| `batch_size` / GPU | 6 |
| `grad_accum_iter` | 12（每 12 次 micro-step 做一次 `optimizer.step`） |
| 有效 batch | **576**（≈600） |
| `max_iter` | 30000 |
| `checkpoint.save_iter` | 1000 |
| 输出目录 | `checkpoints/cosmos_predict2_2b_480p_vla4desk_franka_135_demos_lora__8x3090_bs6_ga12/` |
| W&B | project=`cosmos-policy`，group=`cosmos_v2_finetune`，run name 同 `job.name` |
| 并行 | DDP（`trainer.distributed_parallelism=ddp`，`fsdp_shard_size=1`） |
| LR 调度 | Franka 实验 `cycle_lengths=[10000, ...]`，30k step 时约 10k 后进入第二段 LR |

训练前：`wandb login`（或 `export WANDB_API_KEY=...`）。不需要 W&B 时加 `job.wandb_mode=disabled`。纯训练不必加 `--group libero`（仅跑 LIBERO 仿真评估时需要）。

显存：8×3090 上 `bs=6` + LoRA rank32 通常每卡约 15–20GB 量级；若 OOM 将 `batch_size` 降为 4、`grad_accum_iter` 改为 18（仍保持 \(4\times8\times18=576\)）。

### 单卡 16GB LoRA 微调（有效 batch 192）

适用：**1×16GB**（如 RTX 5070 Ti）。**LoRA rank/alpha=32**（实验默认），**每 GPU `batch_size=3`**，**`grad_accum_iter=64`** → 有效 batch **192**，**`lr=1e-5`**，**每 100 step 存盘**，**W&B 在线记录**。

\[
B_{\text{eff}} = 3 \times 1 \times 64 = 192
\]

```bash
cd /path/to/cosmos-policy

# 可选：减轻显存碎片 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --extra cu128 --python 3.10 \
  torchrun --nproc_per_node=1 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_vla4desk_franka_135_demos_lora" \
  trainer.distributed_parallelism=ddp \
  model.config.fsdp_shard_size=1 \
  ckpt_type=dcp \
  dataloader_train.batch_size=3 \
  trainer.grad_accum_iter=64 \
  trainer.max_iter=30000 \
  checkpoint.save_iter=100 \
  trainer.logging_iter=10 \
  optimizer.lr=1e-5 \
  job.wandb_mode=online \
  job.project=cosmos-policy \
  job.group=cosmos_v2_finetune \
  job.name=cosmos_predict2_2b_480p_vla4desk_franka_135_demos_lora__1x16gb_bs3_ga64_lr1e-5
```

| 项 | 本配方 |
|----|--------|
| GPU | 1×16GB |
| `batch_size` | 3 |
| `grad_accum_iter` | 64（每 64 个 micro-step 一次 `optimizer.step`） |
| 有效 batch | **192** |
| `optimizer.lr` | **1e-5** |
| `checkpoint.save_iter` | **100** |
| `max_iter` | 30000（可按需改 `trainer.max_iter`） |
| 输出目录 | `checkpoints/cosmos_predict2_2b_480p_vla4desk_franka_135_demos_lora__1x16gb_bs3_ga64_lr1e-5/` |
| W&B | `job.wandb_mode=online`，project=`cosmos-policy`，run name 同 `job.name` |

训练前执行 `wandb login`。**勿写 `model=policy_ddp`**（见上一节说明）。

**显存**：`bs=3` + LoRA 在 16GB 上可能接近上限；若 OOM，先试 `dataloader_train.num_workers=0`，或将 `batch_size=2`、`grad_accum_iter=96`（仍保持 \(2\times96=192\)）。

**日志自检**：`total num parameters` 约 **2.29×10⁷**；应有 `LoRA injection successful` 与 `Loading consolidated pretrained weights from: ...LIBERO...`。

### 单卡调试（可选）

```bash
torchrun --nproc_per_node=1 ... \
  dataloader_train.batch_size=2 \
  trainer.grad_accum_iter=1 \
  trainer.max_iter=100 \
  job.wandb_mode=disabled \
  job.name=cosmos_predict2_2b_480p_vla4desk_franka_135_demos_lora__smoke
```

### 全参数微调（可选，显存大）

```bash
experiment="cosmos_predict2_2b_480p_vla4desk_franka_135_demos" \
  trainer.grad_accum_iter=12 \
  dataloader_train.batch_size=6
```

### 断点续训与修改 `max_iter`

检查点写在：

```text
checkpoints/<job.name>/checkpoints/iter_000005000/
checkpoints/<job.name>/checkpoints/latest_checkpoint.txt
```

**已训到 5000 step，只想把上限从 5000 改成 30000，能否接着训？**

可以，需同时满足：

1. **`job.name` 不变**（与 5000 step 那次完全相同）。
2. **只改** `trainer.max_iter=30000`（或更大）；不要改 `batch_size`、`grad_accum_iter`、LoRA rank 等，否则优化器状态与数据分布不一致。
3. 目录里已有 `latest_checkpoint.txt` 且指向 `iter_000005000`（或你想续的 iter）。

用**同一条** `torchrun` 命令，仅把 `trainer.max_iter=30000` 写进去再跑即可；会从 **iteration 5000** 继续，直到 30000。日志里应出现 `Resuming ckpt .../iter_000005000` 且 `keys` 含 `model, optim, scheduler, trainer`，**不会**再走 LIBERO `.pt` 预加载。

若改了 `job.name` → 视为新实验，默认从 0 开始（除非手动拷贝旧 `checkpoints/` 目录）。

若把 `max_iter` 改成 **小于** 当前 iteration（例如已 5000 却设 `max_iter=3000`）→ 启动后会立刻结束训练。

**收敛参考**：LIBERO 上 action L1 约 0.01–0.012（见 [LIBERO.md](LIBERO.md)）；小数据集不必强行对齐 1920 有效 batch。

**推理（待完善）**：LoRA 推理实验名 `cosmos_predict2_2b_480p_vla4desk_franka_135_demos_lora__inference_only`；加载训练产生的 DCP 目录 `.../checkpoints/iter_XXXXX/model/`（非单文件 `.pt`）。

## 推理

（待补充 — 加载 checkpoint 进行真机推理部署）
