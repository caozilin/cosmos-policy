# VLA4Desk: Franka 真机数据训练

与上游环境说明一致，请以 [SETUP.md](SETUP.md)、[README.md](README.md) 为准。本文只补充 Franka / VLA4Desk 相关路径与命令。

## 系统环境

- GPU: RTX 3090 (24GB VRAM) 或云端 4090 等；16G 卡需 T5 CPU/offload
- 主模型: Cosmos Policy Predict2-2B（推理 ~6-9GB）
- 文本编码器: T5-11B **Encoder**（仅预处理；`bf16` 约 **10–12GB** 显存；`fp32` 约 **20–22GB**）

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

真机采集数据路径（按本机修改）: `/media/czl/sata/franka_my_code/vla4desk/collected`

包含 12 个 task，共 135 条唯一指令:
- `banana_on_bowl`, `banana_on_plate`
- `pear_on_bowl`, `pear_on_cup`, `pear_on_plate`
- `strawberry_on_bowl`, `strawberry_on_cup`, `strawberry_on_place`
- `stack_blocks`, `simple_pick_place`
- `fruits_in_contrainers`, `final_clean`

### 提取唯一指令（prompt）

`scripts/extract_vla4desk_prompts.py` 会遍历采集目录下所有 `epo_*/data.json`，汇总去重后的 `prompt`（及对应 `task_name`）。用于转换前核对语言指令、统计条数，或单独生成 `prompts.txt`。

脚本内默认数据根目录为 `DATA_ROOT`（见文件顶部，按本机改为你的 `collected` 路径）。

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

```bash
# 仅 eval → vla4desk/prompts.txt
python scripts/extract_eval_prompts.py \
  --eval-root /home/czl/桌面/毕设/结题报告/素材/eval \
  -o ./vla4desk/prompts.txt

# 采集 135 条 + eval 新增，合并去重（推荐，当前仓库 prompts.txt 约 210 条）
python scripts/extract_eval_prompts.py \
  --merge-committed-prompts \
  --eval-root /home/czl/桌面/毕设/结题报告/素材/eval \
  -o ./vla4desk/prompts.txt
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
  --input /media/czl/sata/franka_my_code/vla4desk/collected \
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

**必须**设 **`HF_HUB_CACHE`** 指向 `hf_cache` 根目录（不要只设 `HF_HOME`）。缓存齐全时设 `HF_HUB_OFFLINE=1`，避免误从网络下载 `spiece.model`。

**常见问题：**

- `TypeError: not a string`（加载 tokenizer）：未设 `HF_HUB_CACHE`，或缓存不完整。
- 进度条 `0/135` 很久不动、GPU 空：正在从磁盘读 ~43G 权重到 CPU；第一次完成后会变为 `1/135`。
- `snapshots/` 下有空目录（无 `spiece.model`）：删除空 snapshot，只保留完整 hash 目录。

```bash
source .venv/bin/activate
export HF_HUB_CACHE=/path/to/cosmos-policy/hf_cache
export HF_HUB_OFFLINE=1

# 推荐：24G+ / 4090，整 Encoder 上 GPU（bf16，默认）
python scripts/precompute_t5_embeddings.py \
  --gpu 0 --device cuda \
  --hf-hub-cache "$HF_HUB_CACHE" --local-files-only \
  -i ./vla4desk/prompts.txt \
  -o ./datasets/VLA4Desk-Franka/success_only/vla4desk_franka/t5_embeddings.pkl

# 16G 显存：GPU 限 12G + CPU offload
python scripts/precompute_t5_embeddings.py --gpu 0 --device auto --max_gpu_mem_gib 12

# fp32 加载（更占显存；输出 pkl 仍为 bf16）
python scripts/precompute_t5_embeddings.py --gpu 0 --device cuda --dtype fp32

# 纯 CPU（不占显存，较慢）
python scripts/precompute_t5_embeddings.py --gpu '' --device cpu
```

脚本参数：`--hf-hub-cache`、`--local-files-only`、`--dtype {bf16,fp32}`（默认 `bf16`）。

也可用 `uv run`（无需手动 activate）：

```bash
export HF_HUB_CACHE=/path/to/cosmos-policy/hf_cache
export HF_HUB_OFFLINE=1
uv run --extra cu128 --python 3.10 python scripts/precompute_t5_embeddings.py \
  --gpu 0 --device cuda --hf-hub-cache "$HF_HUB_CACHE" --local-files-only \
  -i ./vla4desk/prompts.txt \
  -o ./datasets/VLA4Desk-Franka/success_only/vla4desk_franka/t5_embeddings.pkl
```

> 预计算使用 `T5EncoderModel`（仅 Encoder）。`bf16` 显存约 **10–12GB**；磁盘 `pytorch_model.bin` 为 fp32（~43G），读入时 **CPU 内存** 仍可能暂时很高，不等于 45G 全在 GPU 上。

## 离线验证（可选，非训练必须）

`scripts/validate_vla4desk_libero_inference.py`：从转换后的 HDF5 读图 / proprio / action，用 **LIBERO 预训练权重** 做若干时刻的推理，检查数据管线是否正常、真机图能否喂进模型。推理语言条件默认随机抽一条 LIBERO 的 T5 embedding，**不**代表真机任务效果。

须安装 LIBERO 依赖（`--group libero`）。优先使用本地 `hf_cache`（`export HF_HOME=...`，脚本会自动解析快照路径，避免重复下载）。

```bash
export HF_HOME=/path/to/cosmos-policy/hf_cache

uv run --extra cu128 --group libero --python 3.10 \
  python scripts/validate_vla4desk_libero_inference.py \
  --hdf5-path datasets/VLA4Desk-Franka/success_only/vla4desk_franka/put_the_yellow_cube_on_the_red_plate_demo.hdf5 \
  --timestamps-json /path/to/vla4desk/collected/simple_pick_place/epo_1/data.json \
  --output-dir ./validation_outputs/vla4desk_epo_1_hdf5
```

输出：`validation_*.json`、`states_actions_*.csv`、以及 `comparisons/` 下输入图与预测 future 图的对比 PNG（默认 0–18 s、步长 2 s）。

## 训练

（待补充 — 需要根据具体数据格式配置 dataset 和训练脚本）

## 推理

（待补充 — 加载 checkpoint 进行真机推理部署）
