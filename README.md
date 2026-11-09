# PRC：热层大气质量密度经验基线残差校正

**Thermospheric Mass Density Forecasting via PatchTST Residual Correction of an Empirical Baseline**

[中文说明](#中文说明) | [English documentation](#english-documentation)

## 中文说明

### 项目概述

这是从研究项目提取的精简独立发布候选，保留训练、增量微调、推理、绘图、数据处理和架构对比能力。不依赖旧 `karman` 包，不包含论文草稿、专利、历史日志或海量逐窗口预测文件。

本发布包包含研究代码、两份 2024 年 5 月的真实融合数据子集和一份模型权重，可用于运行下文的训练与推理演示。参见 [许可说明](LICENSE_STATUS.md) 和 [发布检查清单](docs/RELEASE_CHECKLIST.md)。

### 环境配置

推荐 Python 3.10。以下命令在项目根目录执行，可用于 PowerShell 或终端：

```sh
conda create -n prc python=3.10 pip -y
conda activate prc
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python tests/check_release.py
python tests/smoke_test.py
```

也可先执行 `conda env create -f environment.yml`，激活环境后单独安装 PyTorch。

GPU 训练请使用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)，依据驱动安装匹配的 CUDA wheel，并将运行配置中的 `device` 改为 `cuda`。不要混用上述 CPU 安装方式和不同 CUDA wheel。

依赖范围是发布候选的版本约束，不代表所有组合均已验证。实际验证版本见 [验证记录](VALIDATION.md) 和 [环境版本快照](environment-tested.json)。已在既有研究环境中完成运行检查，**尚未验证全新环境安装**；环境快照不是稳定版本锁文件。

### 跑通与使用

```sh
python run.py configs/demo_train.json
python run.py configs/demo_infer.json
python scripts/training/infer_plotting.py --help
python scripts/training/finetune_forecasting_transformer.py --help
python scripts/input_data_prep/process_merge_sw_and_tudelft_data_simple.py --help
python experiments/compare_architectures.py --data data/merged_datasets/satellites_data_swarm_c_10xresample_omni.csv --start 2024-05-01 --end 2024-05-31
```

- `run.py` 固定子进程工作目录为项目根目录。从其他目录调用时，请使用启动脚本和配置文件的绝对路径。
- `demo_train` 使用一个月数据训练 1 轮，保留 288 步历史窗口、288 步预测窗口和 64 维模型；这是流程演示，不是完整论文训练方案。CPU 运行较慢，可使用 GPU。
- `demo_infer` 使用可信主权重进行短时间回溯推理。需要足够的历史与目标数据；若有效窗口不足，应扩大可用数据范围。
- 新结果写入 `outputs/`；历史参考指标保留在 `results/reference/`。二者不要混淆。
- 架构对比脚本默认仅预览十组命令，添加 `--execute` 才开始训练。对比时显式指定模型类型、训练目标、模型维度、数据时间范围及 `test_gap_days=0`，保持相同数据划分，并报告各模型专属参数。
- 独立的全年架构对比使用 128 维配置，需要补齐 2024 全年数据。五月子集对照不能作为全年对照结果。

### 目录与数据

| 路径 | 内容 |
|---|---|
| `scripts/training/` | 主训练、微调、推理、绘图与评价；保留原类名以兼容已有权重 |
| `scripts/input_data_prep/` | Swarm、太阳指数和 OMNI 的下载及融合处理链 |
| `experiments/` | 五种模型在两种训练目标下的对比入口 |
| `configs/` | 可执行演示配置，不代表历史实验的完整运行记录 |
| `data/` | 本地候选中的真实一个月融合数据、字段表和来源说明 |
| `models/` | 本地候选中的一份原始主权重，无重复历史权重目录 |
| `results/reference/` | 三年与磁暴事件汇总，不含海量逐样本 CSV |
| `tests/` | 入口检查、数据加载、单批次训练和主权重推理检查 |
| `SOURCE_MANIFEST.json` | 原代码来源、数据筛选规则和权重哈希 |
| `SHA256SUMS.txt` | 发布文件校验清单 |

真实数据子集包含 Swarm-A 和 Swarm-C 的 2024 年 5 月数据，各 8,920 行。数据保留原始数值，**不能替代多年训练数据或支撑完整跨年份复现**。

### 可复现性与发布边界

主模型使用 64 维配置，独立架构对比使用 128 维配置，不能把二者统一改写为同一个历史实验。PatchTST 不使用未来显式条件输入，但最终密度重建仍使用目标时段的经验模型基线，因此回溯评价不自动等同于起报信息完全可得的实时预报。只加载可信来源的权重。

详细说明见 [可复现性与科学边界](docs/REPRODUCIBILITY.md)；数据来源与重建见 [数据说明](docs/DATA.md)。

## English documentation

### Overview

PRC is a compact, standalone release candidate extracted from the research project. It retains training, incremental fine-tuning, inference, plotting, data preparation, and architecture comparisons. It does not depend on the legacy `karman` package and excludes manuscript drafts, patents, historical logs, and large per-window prediction files.

This release includes the research code, two merged May 2024 data subsets, and the model checkpoint used by the training and inference demonstrations. See the [license notice](LICENSE_STATUS.md) and [release checklist](docs/RELEASE_CHECKLIST.md).

### Environment setup

Python 3.10 is recommended. Run these commands from the project root in PowerShell or a terminal:

```sh
conda create -n prc python=3.10 pip -y
conda activate prc
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python tests/check_release.py
python tests/smoke_test.py
```

Alternatively, run `conda env create -f environment.yml`, activate the environment, and install PyTorch separately.

For GPU training, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose a CUDA wheel compatible with your driver, then set `device` to `cuda` in the runtime configuration. Do not mix the CPU installation above with incompatible CUDA wheels.

The dependency ranges are release-candidate constraints, not a guarantee that every version combination has been tested. See the [Validation record](VALIDATION.md) and [Environment snapshot](environment-tested.json) for the versions actually used. Runtime checks passed in an existing research environment; **installation in a clean environment has not yet been validated**. The snapshot is not a stable-version lockfile.

### Quick start and usage

```sh
python run.py configs/demo_train.json
python run.py configs/demo_infer.json
python scripts/training/infer_plotting.py --help
python scripts/training/finetune_forecasting_transformer.py --help
python scripts/input_data_prep/process_merge_sw_and_tudelft_data_simple.py --help
python experiments/compare_architectures.py --data data/merged_datasets/satellites_data_swarm_c_10xresample_omni.csv --start 2024-05-01 --end 2024-05-31
```

- `run.py` sets the subprocess working directory to the project root. When invoking it from another directory, use absolute paths to both the launcher and configuration file.
- `demo_train` trains for one epoch on a one-month subset, retaining a 288-step history, a 288-step forecast horizon, and a 64-dimensional model. It demonstrates the workflow, not the full manuscript training protocol. CPU execution can be slow; GPU execution is supported.
- `demo_infer` performs a short retrospective evaluation using the trusted main checkpoint. Sufficient historical and target data are required; expand the available data range if too few valid windows remain.
- New outputs are written to `outputs/`. Historical reference metrics are stored separately in `results/reference/`; they are not newly reproduced results.
- The architecture-comparison script previews ten commands by default. Add `--execute` to start training. Explicitly specify the architecture, training objective, model dimension, date range, and `test_gap_days=0`; retain identical data splits and report architecture-specific settings.
- The separate full-year architecture comparison uses a 128-dimensional configuration and requires the complete 2024 data. Comparisons on the May subset must not be presented as full-year results.

### Repository layout and data

| Path | Contents |
|---|---|
| `scripts/training/` | Training, fine-tuning, inference, plotting, and evaluation; original class names retained for checkpoint compatibility |
| `scripts/input_data_prep/` | Download and fusion utilities for Swarm, solar indices, and OMNI |
| `experiments/` | Comparisons of five architectures under two training objectives |
| `configs/` | Executable demonstration configurations, not complete historical experiment records |
| `data/` | Real one-month merged subsets in the local candidate, column schema, and provenance information |
| `models/` | One original main checkpoint in the local candidate, without duplicated historical checkpoints |
| `results/reference/` | Three annual summaries and storm-event summaries, excluding large per-sample CSV files |
| `tests/` | Entry-point, data-loading, single-batch training, and main-checkpoint inference checks |
| `SOURCE_MANIFEST.json` | Source-code provenance, data-selection rules, and checkpoint hash |
| `SHA256SUMS.txt` | Release-file checksums |

The real subsets contain May 2024 data from Swarm-A and Swarm-C, with 8,920 rows per satellite. Original numeric values are retained. **These subsets do not replace the multiyear training data or enable full cross-year reproduction.**

### Reproducibility and publication boundaries

The main model uses a 64-dimensional configuration, whereas the separate architecture comparison uses 128 dimensions. These must not be rewritten as one identical historical experiment. PatchTST does not use the explicit future-condition input, but physical-density reconstruction still uses the empirical baseline for the target interval. Retrospective evaluation therefore does not automatically establish real-time forecasting with information available at issuance. Load checkpoints only from trusted sources.

See [Reproducibility and scientific boundaries](docs/REPRODUCIBILITY.md) and [Data provenance and reconstruction](docs/DATA.md).
