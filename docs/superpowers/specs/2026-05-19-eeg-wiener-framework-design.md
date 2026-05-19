# EEG Background Seizure — Wiener 特征工程框架设计规范

**日期**：2026-05-19  
**状态**：已批准  
**范围**：特征工程 pipeline（Wiener 分解 + ICA 对照）+ 可视化；暂不含模型训练

---

## 1. 背景与目标

基于 TUEP v3.1.0 数据集，研究物理点源 Wiener 分解对背景 EEG 特异性成分提取的有效性。本框架聚焦于：

1. 从原始 EDF 文件提取有效背景 epoch 并缓存
2. 实现向量 Wiener 滤波器（核心方法）及固定标量消融对照
3. 实现 FastICA 对照特征工程
4. 物理验证实验 V1/V2/V3
5. 所有结果的可视化输出

模型训练（SVM / XGBoost / CNN）预留接口，后续单独迭代。

---

## 2. 技术决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 框架形式 | 模块化 Python 包 + 脚本 + Notebook | 可复用、可测试、兼顾探索与批量运行 |
| 配置管理 | YAML + CSV 结果日志 | 零依赖，可复现 |
| 中间结果 | 磁盘缓存 `.npz`（项目目录内） | 36GB 数据集，避免重复 IO |
| 测试策略 | pytest + 合成数据 fixture | 不依赖真实数据，任意机器可运行 |
| CNN 框架 | 暂缓（PyTorch 未安装） | 当前阶段聚焦特征工程 |

---

## 3. 项目结构

```
D:\eeg_background_seizure\
│
├── eeg_bg/                          ← 核心包（pip install -e .）
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py              ← 加载 YAML、管理路径常量
│   │
│   ├── io/
│   │   ├── __init__.py
│   │   ├── dataset.py               ← 目录遍历、subject/session 索引
│   │   ├── edf_reader.py            ← EDF 读取、重采样、通道筛选
│   │   ├── annotation.py            ← csv_bi 解析、bckg 段提取
│   │   └── cache.py                 ← npz 读写、缓存键哈希
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── epoch.py                 ← 8s epoch 切割、伪迹剔除
│   │   └── reference.py             ← AR/LE 检测与筛选
│   │
│   ├── decomposition/
│   │   ├── __init__.py
│   │   ├── wiener.py                ← 向量 Wiener 滤波器（核心）
│   │   ├── wiener_scalar.py         ← 固定标量模式（消融对照）
│   │   └── ica.py                   ← FastICA + 自动伪迹检测
│   │
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── coherence.py             ← V1：分解纯净性验证
│   │   └── transitivity.py          ← V2/V3：传递性约束、频率相关性
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── coherence_plots.py       ← 相干度热图、降幅箱线图
│       ├── filter_plots.py          ← |h(f)| 和 ∠h(f) 曲线
│       └── verification_plots.py    ← V1/V2/V3 汇总图
│
├── tests/
│   ├── conftest.py                  ← 合成 EEG fixtures、临时 cache 目录
│   ├── test_io/
│   │   ├── test_dataset.py
│   │   ├── test_edf_reader.py
│   │   ├── test_annotation.py
│   │   └── test_cache.py
│   ├── test_preprocessing/
│   │   ├── test_epoch.py
│   │   └── test_reference.py
│   ├── test_decomposition/
│   │   ├── test_wiener.py
│   │   └── test_ica.py
│   └── test_verification/
│       ├── test_coherence.py
│       └── test_transitivity.py
│
├── scripts/
│   ├── 01_extract_epochs.py         ← 提取 bckg epoch 并缓存
│   ├── 02_run_wiener.py             ← 批量 Wiener 分解并缓存
│   ├── 03_run_ica.py                ← 批量 ICA 并缓存
│   └── 04_run_verification.py       ← V1/V2/V3 物理验证
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_wiener_demo.ipynb
│   ├── 03_ica_comparison.ipynb
│   └── 04_verification_results.ipynb
│
├── configs/
│   └── default.yaml
│
├── cache/                           ← 自动生成，git ignored
│   ├── epochs/
│   ├── wiener/
│   └── ica/
│
├── results/                         ← 图表输出，git ignored
├── docs/
├── setup.py
└── requirements.txt
```

### 数据流向（单向无环）

```
EDF 文件
  → [io] 读取 + 标注解析
  → [preprocessing] epoch 切割 + 伪迹剔除
  → [cache/epochs]           ← 缓存检查点 1
  → [decomposition/wiener|ica] 分解
  → [cache/wiener|ica]       ← 缓存检查点 2
  → [verification] 指标计算
  → [visualization] 图表 → results/
```

---

## 4. 配置文件 `configs/default.yaml`

```yaml
paths:
  data_root: "D:/EEGdata/TUEP/v3.1.0"
  cache_dir: "cache"
  results_dir: "results"

dataset:
  reference_scheme: "ar"
  montage_dir: "01_tcp_ar"
  classes:
    epilepsy: "00_epilepsy"
    control:  "01_no_epilepsy"

split:
  train: 0.70
  val:   0.10
  test:  0.20
  random_seed: 42

preprocessing:
  target_sfreq: 125
  bandpass: [0.5, 40.0]
  epoch_length_sec: 8.0
  artifact_threshold_uv: 200.0
  seizure_buffer_sec: 30.0

channels:
  standard_19: ["FP1","FP2","F3","F4","F7","F8",
                "T3","T4","T5","T6","P3","P4",
                "O1","O2","Fz","Cz","Pz","C3","C4"]
  bilateral_pairs:
    - ["FP1", "FP2"]
    - ["F3",  "F4"]
    - ["F7",  "F8"]
    - ["T3",  "T4"]
    - ["T5",  "T6"]
    - ["P3",  "P4"]
    - ["O1",  "O2"]
  midline: ["Fz", "Cz", "Pz"]

wiener:
  mode: "frequency"
  nperseg: 1000
  freq_resolution_hz: 0.5
  coherence_threshold: 0.15
  freq_band: [0.5, 40.0]

ica:
  n_components: 19
  method: "fastica"
  artifact_corr_threshold: 0.8

verification:
  v2_transitivity_amp_threshold: 0.1
  v2_transitivity_phase_threshold: 0.392
  v3_freq_variation_threshold: 0.20
```

---

## 5. 核心数据结构

```python
@dataclass
class EpochRecord:
    subject_id:   str
    session_id:   str
    token_id:     str
    label:        int          # 0=epilepsy, 1=control
    reference:    str          # "ar" | "le"
    split:        str          # "train" | "val" | "test"
    edf_path:     Path
    start_sec:    float
    cache_key:    str          # sha256(edf_path + start_sec + cfg)[:16]

@dataclass
class WienerResult:
    subject_id:    str
    epoch_idx:     int
    raw:           np.ndarray   # (n_channels, n_times)
    specific:      np.ndarray   # (n_channels, n_times)
    coherent:      np.ndarray   # (n_channels, n_times)
    filters:       dict         # {pair_key: h(f) array (n_ref, n_freqs)}
    freqs:         np.ndarray   # 频率轴
    skipped_pairs: list[str]    # 低相干度跳过的导联对

@dataclass
class VerificationReport:
    subject_id:        str
    v1_coherence_pre:  np.ndarray  # (n_ch, n_ch) 分解前
    v1_coherence_post: np.ndarray  # (n_ch, n_ch) 分解后
    v2_amp_errors:     np.ndarray  # 传递性幅度误差
    v2_phase_errors:   np.ndarray  # 传递性相位误差
    v3_freq_variation: np.ndarray  # |h(f)| 频段方差
```

**缓存键设计：**
```python
cache_key = sha256(f"{edf_path}|{start_sec}|{sfreq}|{bandpass}").hexdigest()[:16]
# 路径：cache/epochs/{subject_id}/{cache_key}.npz
```

---

## 6. 函数签名

### 6.1 `eeg_bg/io/`

```python
# dataset.py
def build_subject_index(cfg: dict) -> pd.DataFrame: ...
def assign_splits(index: pd.DataFrame, cfg: dict) -> pd.DataFrame: ...

# edf_reader.py
def load_edf(edf_path: Path, cfg: dict) -> tuple[np.ndarray, list[str], float]: ...

# annotation.py
def extract_bckg_intervals(csv_bi_path: Path, cfg: dict) -> list[tuple[float, float]]: ...

# cache.py
def load_or_compute(cache_path: Path, compute_fn: Callable, force_recompute: bool = False) -> np.ndarray: ...
def make_cache_key(edf_path: Path, start_sec: float, cfg: dict) -> str: ...
```

### 6.2 `eeg_bg/preprocessing/`

```python
# epoch.py
def slice_epochs(data, sfreq, intervals, epoch_len_sec, artifact_threshold_uv) -> np.ndarray: ...
def bandpass_filter(data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray: ...

# reference.py
def detect_reference(ch_names: list[str], montage_dir: str) -> str: ...
def filter_by_reference(index: pd.DataFrame, scheme: str) -> pd.DataFrame: ...
```

### 6.3 `eeg_bg/decomposition/wiener.py`

```python
def estimate_cross_psd(data, sfreq, nperseg) -> tuple[np.ndarray, np.ndarray]: ...
def compute_wiener_filter(S, target_idx) -> np.ndarray: ...
def apply_wiener_filter(data, h, ref_indices, sfreq) -> tuple[np.ndarray, np.ndarray]: ...
def decompose_epoch(epoch, ch_names, cfg) -> WienerResult: ...
def decompose_subject(epochs, ch_names, subject_id, cfg) -> list[WienerResult]: ...
```

### 6.4 `eeg_bg/decomposition/ica.py`

```python
def fit_ica(epochs, cfg) -> tuple[Any, list[int]]: ...
def apply_ica(epochs, ica_model, artifact_indices) -> np.ndarray: ...
```

### 6.5 `eeg_bg/verification/`

```python
# coherence.py
def compute_pairwise_coherence(data, sfreq, nperseg, freq_band) -> np.ndarray: ...
def run_v1(results, cfg) -> pd.DataFrame: ...

# transitivity.py
def run_v2(results, cfg) -> pd.DataFrame: ...
def run_v3(results, cfg) -> pd.DataFrame: ...
```

### 6.6 `eeg_bg/visualization/`

```python
# filter_plots.py
def plot_wiener_filter_response(result, pair_key, ax=None) -> plt.Figure: ...
def plot_all_pairs_response(result, save_path=None) -> plt.Figure: ...

# coherence_plots.py
def plot_coherence_matrix(coh_pre, coh_post, ch_names, title="") -> plt.Figure: ...
def plot_coherence_reduction(v1_df, group_by="pair") -> plt.Figure: ...
def plot_signal_decomposition(result, ch_name, epoch_idx=0, time_window=(0,4)) -> plt.Figure: ...

# verification_plots.py
def plot_v2_transitivity(v2_df) -> plt.Figure: ...
def plot_v3_frequency_variation(v3_df) -> plt.Figure: ...
def plot_ica_vs_wiener_coherence(wiener_post, ica_post, raw_pre, ch_names) -> plt.Figure: ...
```

**输出约定**：所有函数返回 `plt.Figure`，不调用 `plt.show()`，调用方负责保存。

---

## 7. 测试框架

### 测试策略
- **合成数据优先**：`conftest.py` 生成已知点源模型的合成 EEG，不依赖真实数据
- **单元测试**：每个函数独立可测
- **集成测试**：`@pytest.mark.integration` 标记，需真实 EDF，平时跳过

### 关键 Wiener 测试用例
合成信号 = 已知点源 × 已知传输函数 + 独立噪声，验证：
1. 分解后特异性成分与点源相干度 < 0.05
2. `|h_ij(f)|` 估计误差 < 5%
3. 传递性约束误差 ε < 0.1

### 运行命令
```bash
pytest tests/ -v                     # 全部单元测试
pytest tests/ -m "not integration"   # 跳过集成测试（默认）
pytest tests/test_decomposition/     # 只测某模块
```

---

## 8. Notebook 结构

| Notebook | 主要内容 |
|----------|----------|
| `01_data_exploration` | 数据集统计、EDF 波形预览、epoch 数量分布 |
| `02_wiener_demo` | 单被试 Wiener 分解、三行波形图、滤波器响应、相干度热图 |
| `03_ica_comparison` | ICA vs Wiener 三列相干度对比图 |
| `04_verification_results` | V1/V2/V3 汇总图表、按组分列统计表 |

---

## 9. 范围边界

**本框架包含：**
- 数据加载、预处理、epoch 提取、缓存
- Wiener 分解（频率相关 + 固定标量消融）
- ICA 对照
- 物理验证 V1/V2/V3
- 全套可视化

**本框架不包含（后续迭代）：**
- 手工特征提取（频带功率、Hjorth、谱熵）
- SVM / XGBoost 训练与评估
- 浅层 CNN 训练（需先安装 PyTorch）
- Grad-CAM / SHAP 可解释性分析
