# Lab2: SimCLR 对比实验复现指南

## 1. 项目概览
- 目标：在 CIFAR-10 / STL-10 等数据集上比较 **SimCLR 自监督学习**、**自编码器 (AE)** 与 **监督学习** 的表征能力。
- 主要训练流程：
  1. 自监督预训练（SimCLR、AutoEncoder）。
  2. 线性评估与全模型微调。
  3. 监督基线训练。
- 目录结构：
  - `data/`：数据集与增强定义。
  - `models/`：ResNet 编码器、AE、投影头、分类头。
  - `losses/`：SimCLR 损失等。
  - `scripts/`：批量实验脚本（bash 版本）。
  - 顶层 `*.py`：单次实验入口（`simclr_pretrain.py`、`supervised_train.py` 等）。

## 2. 环境配置
1. 准备 Python ≥ 3.9（推荐 Conda 或 Miniconda）：
   ```bash
   conda create -n lab2 python=3.9
   conda activate lab2
   ```
2. 安装 PyTorch 与常用依赖（根据自身 CUDA 版本选择对应命令）：
   ```bash
   # 以 CUDA 11.8 为例
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install numpy scipy scikit-learn tqdm pandas
   ```
   项目中仅依赖常见科学计算库，如需完整列表可运行 `pip install -r requirements.txt`（若你在本地整理了依赖文件）。
3. 准备数据目录：全部脚本默认使用 `./data`，若不存在会自动在首次运行时下载 CIFAR-10 / STL-10。
4. 设置 GPU：bash 脚本支持 `--gpu` 参数，或手动设置 `CUDA_VISIBLE_DEVICES`。

## 3. 单次实验脚本
每个入口都支持 `--help` 查看全部参数，下面列出常用用法：

| 脚本 | 作用 | 示例 |
| --- | --- | --- |
| `python supervised_train.py` | 从零训练 ResNet 编码器和分类头，支持缺失标签比例。 | `python supervised_train.py --dataset cifar10 --backbone resnet18 --label-fraction 0.15 --experiment-name cifar10_resnet18_supervised_few` |
| `python simclr_pretrain.py` | 运行 SimCLR 预训练，支持增广、损失温度等。 | `python simclr_pretrain.py --dataset cifar10 --backbone resnet18 --augment full --loss nt_xent --epochs 100 --experiment-name cifar10_resnet18_simclr_full` |
| `python linear_eval.py` | 冻结 encoder 进行线性探针评估。 | `python linear_eval.py --dataset cifar10 --backbone resnet18 --encoder-checkpoint checkpoints/xxx/simclr_encoder.pt --projection-checkpoint checkpoints/xxx/simclr_projection.pt --proj-mode head_on_hfeat` |
| `python finetune.py` | 在下游任务上微调 SimCLR/AE 编码器。 | `python finetune.py --dataset cifar10 --encoder-type resnet --encoder-checkpoint checkpoints/xxx/simclr_encoder.pt --label-fraction 0.15` |
| `python ae_pretrain.py` | 训练 AutoEncoder 作为无监督基线。 | `python ae_pretrain.py --dataset cifar10 --latent-dim 128 --experiment-name cifar10_resnet18_ae_full` |

所有结果（日志/指标/权重）会写入 `results/<experiment-name>/` 与 `checkpoints/<experiment-name>/`。

## 4. 批量实验脚本
`scripts/` 目录提供了常用流程的 bash 模板，可批量复现实验。通用参数：`--dataset`、`--backbone`、`--gpu`。

- `scripts/run_main_full_labels.sh`：
  1. 监督基线 (100% 标签)；
  2. SimCLR 预训练 → 线性评估 → 全模型微调；
  3. AE 预训练 → AE 微调。
  ```bash
  bash scripts/run_main_full_labels.sh --gpu 0 --dataset cifar10 --backbone resnet18
  ```
- `scripts/run_main_few_labels.sh`：与上类似，但监督/下游部分仅用 15% 标签。
- `scripts/run_ablation_augment.sh`：遍历 `only_crop / crop_color / crop_color_gray / full` 组合。
- `scripts/run_ablation_loss.sh`：比较 `nt_xent / cosine_margin / pos_only` 损失。
- `scripts/run_ablation_proj_head.sh`：比较 projection head 结构（`no_head`、`head_on_hfeat`、`head_on_zfeat`）。

脚本会依次调用相应的 Python 入口，自动把中间模型保存在 `checkpoints/` 并执行线性评估。

## 5. 流程示例
1. **完整实验（全标签）**：
   ```bash
   conda activate lab2
   bash scripts/run_main_full_labels.sh --gpu 0 --dataset cifar10 --backbone resnet18
   ```
2. **低标签比例实验**：
   ```bash
   bash scripts/run_main_few_labels.sh --gpu 0 --dataset cifar10 --backbone resnet18
   ```
3. **SimCLR 增广消融**：
   ```bash
   bash scripts/run_ablation_augment.sh --gpu 1 --dataset cifar10 --backbone resnet18
   ```
运行结束后，可在 `results/<exp>/` 查看 `*_log.csv`、`*_metrics.json`，在 `checkpoints/<exp>/` 找到 `simclr_encoder.pt`、`simclr_projection.pt`、`ae_encoder.pt` 等。
