# Wiener phase/coherence 实验运行说明

本文说明 `scripts/run_wiener_threshold_phase_experiment.sh` 的手动运行方法。该脚本执行 8 组 Wiener 阈值/相位门控实验，并为每组训练两个 XGBoost 特征 profile。

## 1. 实验矩阵

| 实验 | Wiener 模式 | coherence threshold | phase gate threshold |
|---|---|---:|---:|
| exp1 | frequency | 0.15 | π |
| exp2 | frequency | 0.45 | π |
| exp3 | frequency | 0.75 | π |
| exp4 | phasegated | 0.15 | π/2 |
| exp5 | phasegated | 0.15 | π/5 |
| exp6 | phasegated | 0.15 | π/10 |
| exp7 | phasegated | 0.45 | π/10 |
| exp8 | phasegated | 0.75 | π/10 |

固定项包括随机种子、subject-level split、channel groups、预处理、XGBoost 配置和 `coherence_weighted` overlap fusion。每组都会训练：

- `base211`
- `base211_conn80`

## 2. 运行前检查

在仓库根目录执行：

```bash
cd /root/eeg_background_seizure
conda run -n eeg_pipeline python -c "import eeg_bg, xgboost, shap; print('environment ok')"
bash scripts/run_wiener_threshold_phase_experiment.sh --help
```

确认以下数据已经存在：

```bash
find cache/epochs -name '*.npz' | wc -l
```

如果跳过 epoch 提取，`cache/epochs/` 必须已经完整生成。

脚本会删除配置中对应的 cache/results 子目录，尤其是 `--clear-cache` 会删除整个 `cache_dir`。运行前请确认没有其他任务正在使用这些目录。

## 3. 首次完整运行

脚本默认 worker 数已经是 16。推荐显式写出，便于日志复现：

```bash
bash scripts/run_wiener_threshold_phase_experiment.sh \
  --workers 16
```

首次运行的阶段顺序为：

1. script 01：提取 epochs（只运行一次）；
2. exp1–exp8：运行 script 02 Wiener，并分别运行 script 06 的两个 feature profile；
3. script 03：生成 ICA baseline；
4. 训练 raw/ICA baseline 的两个 feature profile；
5. 将 baseline 复制到每个实验目录；
6. 运行 script 07 归档每个实验；
7. 输出 8 组 raw / ICA / Wiener condition 的 AUROC 摘要。

脚本 04 verification 已被刻意跳过，因此本次归档不会产生 V1、connectivity、gate 或 skipped-pairs 验证结果。每个实验开始时会清理该实验的 `results/verification/`，避免把旧验证结果误归档。

## 4. 常用运行方式

### 已有 epochs，跳过 script 01

```bash
bash scripts/run_wiener_threshold_phase_experiment.sh \
  --skip-epochs \
  --workers 16
```

### 已有 baseline，跳过 script 03 和 raw/ICA XGBoost

```bash
bash scripts/run_wiener_threshold_phase_experiment.sh \
  --skip-baseline \
  --workers 16
```

此时必须已经存在：

```text
results/exp_wiener_phase/baseline/xgboost/base211/raw/
results/exp_wiener_phase/baseline/xgboost/base211/ica/
results/exp_wiener_phase/baseline/xgboost/base211_conn80/raw/
results/exp_wiener_phase/baseline/xgboost/base211_conn80/ica/
```

### 从某个 Wiener 实验开始

例如从 exp4 开始：

```bash
bash scripts/run_wiener_threshold_phase_experiment.sh \
  --from 4 \
  --skip-epochs \
  --skip-baseline \
  --workers 16
```

注意：`--from N` 只跳过 exp1 到 exp(N-1) 的 Wiener/XGBoost 循环；脚本后面的 baseline 复制和 script 07 归档仍会遍历 exp1–exp8。因此使用断点续跑时，应确认此前实验目录已经有完整结果，或根据需要手工归档单个实验。

### 完全重建 cache

```bash
bash scripts/run_wiener_threshold_phase_experiment.sh \
  --clear-cache \
  --workers 16
```

`--clear-cache` 是破坏性选项，会删除配置中的整个 `cache_dir`，只应在确认可以重新运行 script 01 时使用。

## 5. 运行日志与结果

运行日志写入：

```text
results/exp_wiener_phase/runtime_YYYY-MM-DD_HHMMSS.log
```

每组的中间结果位于：

```text
results/exp_wiener_phase/exp1/
results/exp_wiener_phase/exp2/
...
results/exp_wiener_phase/exp8/
```

XGBoost 结果结构为：

```text
expN/xgboost/base211/{raw,ica,wiener,wiener_phasegated}/
expN/xgboost/base211_conn80/{raw,ica,wiener,wiener_phasegated}/
```

最终归档写入：

```text
experiments/YYYY-MM-DD_HHMMSS_exp_wiener_phase_N_wiener-phase-expN/
```

每个归档包含配置快照、`experiment.json`、`report.md`、各 feature profile 的指标/SHAP 文件。由于 script 04 被跳过，归档中的 `experiment.json` 应显示 `has_verification: false`。

## 6. 失败后的处理

脚本使用 `set -euo pipefail`，任一步骤失败都会停止。建议按日志定位最后一个 `[OK]` 步骤。

常见处理方式：

```bash
# 如果 epochs 已完成，从失败的实验编号继续
bash scripts/run_wiener_threshold_phase_experiment.sh \
  --from N --skip-epochs --workers 16

# 如果 baseline 已完成，同时跳过 baseline 重训
bash scripts/run_wiener_threshold_phase_experiment.sh \
  --from N --skip-epochs --skip-baseline --workers 16
```

如果 script 06 因 feature cache 维度或 schema 不匹配失败，应保留脚本的 `--force` 行为，不要直接复用不确定的旧 feature cache。若 baseline 目录缺少任一 profile 的 raw/ICA 结果，脚本会在复制阶段停止，需要先补齐 baseline。

## 7. 运行完成后的检查

```bash
find experiments -maxdepth 1 -type d -name '*wiener-phase-exp*' | sort | tail
find results/exp_wiener_phase -name test_metrics.json | wc -l
```

预期每个实验每个 profile 包含 3 个 XGBoost condition 的 test metrics：raw、ica 和对应的 Wiener condition；两个 profile 合计应有 6 个 `test_metrics.json`。最后检查终端摘要中的 raw / ICA / Wiener AUROC 是否均为数值而不是 `n/a`。
