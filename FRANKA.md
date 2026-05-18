# VLA4Desk: Franka 真机数据训练

在 [SETUP.md](SETUP.md)、[README.md](README.md) 环境说明基础上，本文说明如何用 **VLA4Desk 采集格式** 的数据微调 Cosmos Policy（LIBERO 预训练 → Franka 演示）。

## 推荐流程

1. **安装**：`uv sync --extra cu128 --python 3.10`（真机训练**不需要** `--group libero`）
2. **放置数据**：自建 `vla4desk/collected/<task>/epo_*/`（仓库内无此目录，见下文）
3. **转 HDF5**：`scripts/convert_vla4desk_to_libero_hdf5.py`
4. **离线预计算 T5**（约 45GB 权重 + ~22GB 显存）：`precompute_t5_embeddings.py` → `t5_embeddings.pkl`
5. **训练**：实验名 `cosmos_predict2_2b_480p_vla4desk_franka_135_demos_lora`（LoRA，推荐）

路径默认相对**仓库根目录**（`cosmos_policy/config/output_paths.py` 会设置 `HF_HOME`、`BASE_DATASETS_DIR`、`checkpoints/` 等，一般无需再写绝对路径）。

## 仓库内目录

**Git 里实际只有** `vla4desk/prompts.txt`（`.gitignore` 规则：`vla4desk/*` 但保留 `*.txt`）。下面带 ※ 的目录/文件需**自行准备**，clone 后默认不存在，也不会被提交。

```text
cosmos-policy/
  vla4desk/
    prompts.txt          # [git] 语言指令示例列表
    collected/           # [本地 ※] 真机原始采集，按 task/epo_* 放置
    eval/                # [本地 ※] 可选；仅在使用 extract_eval_prompts 合并 eval prompt 时需要
  datasets/.../vla4desk_franka/   # [本地 ※] convert 脚本生成
  hf_cache/              # [本地 ※] T5-11b 等权重（约 45GB）
  checkpoints/           # [本地 ※] 训练输出
```

## 环境要求

| 组件 | 说明 |
|------|------|
| GPU | 训练 LoRA：建议 ≥24GB/卡；单卡 16GB 需减小 batch 并加大梯度累积 |
| 主模型 | Cosmos Policy Predict2-2B |
| T5-11B（整包） | 磁盘约 **45GB**；预计算时 Encoder 上 GPU 约 **22GB** 显存（`fp32`），`bf16` 约 10–12GB |
| 训练阶段 | **不加载** T5，只读离线 `t5_embeddings.pkl`（须提前 precompute） |

## 安装

使用 **`uv sync --extra cu128 --python 3.10`**，不要用系统 `pip install -e` 或 Python 3.13 自建 venv。

```bash
cd <仓库根目录>
uv sync --extra cu128 --python 3.10
source .venv/bin/activate
python -V   # 应为 3.10.x
```

- **Docker**：见 [SETUP.md](SETUP.md)；数据在仓库外时挂载到容器内路径（如 `/data`）。
- **LIBERO 仿真评测**：另装 `uv sync --extra cu128 --group libero --python 3.10`（见 [LIBERO.md](LIBERO.md)）。

## 数据格式

### 原始采集（`vla4desk/collected/`，需本地创建）

将 VLA4Desk 采集结果放到 `vla4desk/collected/<task_name>/epo_*/`（路径与 `output_paths.default_vla4desk_collected_dir()` 一致）。每个 episode 含 `cam1.mp4`、`cam2.mp4`、`data.json`。`data.json` 中有 `prompt`、`task_name`、逐帧 `state` / `action` 等。

本仓库示例集（转换脚本会扫全部 task）包含例如：`banana_on_bowl`、`simple_pick_place`、`stack_blocks` 等；转换后得到**按 prompt 分文件**的 HDF5（每个唯一指令一个 `*_demo.hdf5`）。

### 提取 / 合并 prompt

```bash
# 从 collected 汇总唯一 prompt
python scripts/extract_vla4desk_prompts.py
python scripts/extract_vla4desk_prompts.py -o ./vla4desk/prompts.txt

# 可选：从 eval telemetry 合并（须先有目录 vla4desk/eval/.../telemetry.jsonl）
python scripts/extract_eval_prompts.py -o ./vla4desk/prompts.txt
python scripts/extract_eval_prompts.py --merge-committed-prompts -o ./vla4desk/prompts.txt
```

没有 eval 数据时可跳过；`--eval-root` 也可指向任意目录。

## 转换为 LIBERO 风格 HDF5

```bash
python scripts/convert_vla4desk_to_libero_hdf5.py --suite-name vla4desk_franka
```

默认 `--input ./vla4desk/collected`，`--output ./datasets/VLA4Desk-Franka/success_only`。

**与 OpenPI 转换约定一致**：`action` 按 `data.json` 原样写入；帧与视频逐帧对齐；`prompt` 进入文件名与 HDF5 属性 `task_description`。

**输出示例：**

```text
datasets/VLA4Desk-Franka/success_only/vla4desk_franka/
  put_the_banana_in_the_red_bowl_demo.hdf5
  prompts.txt
  t5_embeddings.pkl
  conversion_manifest.json
```

**常用选项：**

```bash
# 256 + JPEG（省空间，与 LIBERO regen 一致）
python scripts/convert_vla4desk_to_libero_hdf5.py --suite-name vla4desk_franka --resize 256 --jpeg-compress

# 跳过 T5 pkl 复制（之后单独 precompute）
python scripts/convert_vla4desk_to_libero_hdf5.py --suite-name vla4desk_franka --skip-t5
```

转换过程**不加载** T5 模型；耗时主要在读 MP4、resize 与写 HDF5。

## T5 embedding 预计算（离线一步，训练前完成）

**为何离线做 pkl：** T5-11b 整包约 **45GB** 磁盘；用 Encoder 做预计算时 GPU 显存约 **22GB**（`fp32`）。Cosmos Policy **训练/推理只读** `t5_embeddings.pkl`，不会在训练 loop 里再加载 T5。因此在一台大显存机器上跑完 `precompute_t5_embeddings.py`，把生成的 `t5_embeddings.pkl` 随 `datasets/.../vla4desk_franka/` 分发即可。

先确保 `hf_cache` 中有完整 `google-t5/t5-11b`（`models--google-t5--t5-11b/snapshots/<hash>/` 含 `pytorch_model.bin`、`spiece.model`、`tokenizer.json`）。

```bash
export HF_HUB_CACHE=./hf_cache
export HF_HUB_OFFLINE=1   # 缓存齐全时使用

python scripts/precompute_t5_embeddings.py \
  --gpu 0 --device cuda \
  --hf-hub-cache "$HF_HUB_CACHE" --local-files-only \
  -i ./datasets/VLA4Desk-Franka/success_only/vla4desk_franka/prompts.txt \
  -o ./datasets/VLA4Desk-Franka/success_only/vla4desk_franka/t5_embeddings.pkl
```

| 场景 | 建议 |
|------|------|
| 24GB+ 显存 | `--device cuda`（默认 `fp32`） |
| 省显存 | `--dtype bf16` |
| 16GB 显存 | `--device auto --max_gpu_mem_gib 12 --dtype bf16` |
| 无 GPU | `--gpu '' --device cpu` |

**常见问题：** 未设 `HF_HUB_CACHE` 会导致 tokenizer 报错；首次加载会把约 45GB 权重读入内存/显存，进度条在 `0/N` 停留较久属正常。

## 离线验证（可选）

需要 LIBERO 依赖组。用 LIBERO 预训练权重对**已转换 HDF5** 做推理对比（非训练必需）：

```bash
export HF_HOME=./hf_cache

uv run --extra cu128 --group libero --python 3.10 \
  python scripts/validate_vla4desk_libero_inference.py \
  --output-dir ./validation_outputs/vla4desk_smoke
```

主要参数均有默认值（示例 HDF5、`t5_embeddings.pkl`、可选 `data.json` 时间戳侧车）。输出 JSON/CSV 与 `comparisons/` 对比图。

## 训练

配置定义：`cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py`。

| 实验名 | 说明 |
|--------|------|
| **`cosmos_predict2_2b_480p_vla4desk_franka_135_demos_lora`** | **推荐**：LoRA 微调 LIBERO 预训练权重 |
| `cosmos_predict2_2b_480p_vla4desk_franka_135_demos` | 全参数微调（显存需求高） |

共同点：

- 数据：`LIBERODataset`，目录 `datasets/.../vla4desk_franka/`
- T5：`同目录/t5_embeddings.pkl`
- 统计量：数据目录内 **`dataset_statistics.json`**（不要用 `libero_dataset_statistics.json`）
- 初始化：`nvidia/Cosmos-Policy-LIBERO-Predict2-2B`（`.pt` 冷启动；有 DCP 断点则自动续训）

### 重要：不要用 `model=policy_ddp` 配 LoRA 实验

Hydra 会用 `policy_ddp` **整段替换** experiment 里的 `model`，导致 `use_lora=False`（变成全参 DDP）。若要多卡 DDP + LoRA，请用：

```text
trainer.distributed_parallelism=ddp
model.config.fsdp_shard_size=1
```

### 多卡 LoRA 示例（8×24GB）

**有效 batch** = `batch_size` × GPU 数 × `grad_accum_iter`  
本例：6 × 8 × 12 = **576**

```bash
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
  optimizer.lr=1e-4 \
  job.name=my_franka_lora_8gpu \
  job.wandb_mode=disabled
```

按需改 `batch_size` / `grad_accum_iter` 以匹配 GPU 数量与显存；OOM 时减小每卡 batch、增大 `grad_accum_iter`，尽量保持有效 batch 不变。

### 单卡 LoRA 示例（16GB 量级）

**有效 batch**：3 × 1 × 64 = **192**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 可选，缓解显存碎片

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
  checkpoint.save_iter=500 \
  optimizer.lr=1e-5 \
  job.name=my_franka_lora_1gpu \
  job.wandb_mode=disabled
```

### 冒烟 / 断点续训

```bash
# 短跑验证环境
torchrun --nproc_per_node=1 ... \
  dataloader_train.batch_size=2 \
  trainer.grad_accum_iter=1 \
  trainer.max_iter=50 \
  job.wandb_mode=disabled \
  job.name=smoke_test
```

检查点：`checkpoints/<job.name>/checkpoints/iter_XXXXXXXX/` 与 `latest_checkpoint.txt`。

**续训**：保持 **`job.name` 不变**，只改 `trainer.max_iter`（或更大）；不要改 batch、LoRA rank 等。日志应出现 `Resuming ckpt`，而不是再次 `Loading consolidated pretrained weights`。

**冷启动自检**（rank0）：`Loading consolidated pretrained weights from: ...LIBERO...pt`、`LoRA injection successful`；不应是 `Training from scratch.`。

W&B：需要时设 `job.wandb_mode=online` 并 `wandb login`；默认示例使用 `disabled`。

### 全参数微调（可选）

```bash
experiment="cosmos_predict2_2b_480p_vla4desk_franka_135_demos" \
  dataloader_train.batch_size=6 \
  trainer.grad_accum_iter=12
```

## 推理

（待补充）训练产物为 DCP 目录 `checkpoints/<job.name>/checkpoints/iter_XXXXX/model/`。推理配置名：`cosmos_predict2_2b_480p_vla4desk_franka_135_demos_lora__inference_only`。
