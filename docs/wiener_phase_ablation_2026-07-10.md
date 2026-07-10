# Wiener 相位门控与重叠通道加权融合实验总结

**数据源：** `experiments/2026-07-10_*_wiener_phase_*/`  
**生成日期：** 2026-07-10  
**模型：** XGBoost + 211 维手工特征  
**指标口径：** 以下性能均为 test subject-level 指标；raw/ICA 是每个归档中重复出现的同一基线结果。

本报告沿用 2026-07-09 报告的结构，并补充本次代码修复后的重叠通道处理分析。需要先说明：本批 8 组实验全部采用 `overlap_policy: coherence_weighted`，没有同一批数据上同时运行“后写覆盖”版本的严格 A/B 对照；因此第 5 节的融合影响分析是**同配置、跨日期的准对照**，不是随机化实验，也不能替代统计显著性检验。

## 1. 实验变量与参数矩阵

8 组实验的 `channel_groups` 完全相同，改变的是 Wiener 模式、相位门控阈值和组级 coherence 阈值。所有组都使用新的目标通道级融合：一个重叠通道若同时收到多个组的 coherent 候选，则按目标-参考 coherence 归一化加权，而不是按组列表顺序覆盖。

| 实验 | 方法 | condition | 相位阈值 | coherence threshold | overlap policy |
|---|---|---|---:|---:|---|
| exp1 | frequency | wiener | π | 0.15 | coherence_weighted |
| exp2 | frequency | wiener | π | 0.45 | coherence_weighted |
| exp3 | frequency | wiener | π | 0.75 | coherence_weighted |
| exp4 | phasegated | wiener_phasegated | π/2 | 0.15 | coherence_weighted |
| exp5 | phasegated | wiener_phasegated | π/5 | 0.15 | coherence_weighted |
| exp6 | phasegated | wiener_phasegated | π/10 | 0.15 | coherence_weighted |
| exp7 | phasegated | wiener_phasegated | π/10 | 0.45 | coherence_weighted |
| exp8 | phasegated | wiener_phasegated | π/10 | 0.75 | coherence_weighted |

重叠通道为：`T3`（G2/G3）、`O1`（G3/G4）、`T4`（G5/G6）、`O2`（G4/G6）。9 个 passthrough 通道仍不经过 Wiener 处理。

## 2. Test 性能结果

### Baseline

| 条件 | Test AUROC | Test Accuracy | Test F1 |
|---|---:|---:|---:|
| raw | 0.7529 | 0.7297 | 0.7247 |
| ica | 0.6382 | 0.5676 | 0.5256 |

### Wiener 参数组

| 实验 | 相位 | coh | Test AUROC | Test Accuracy | Test F1 | AUROC - raw |
|---|---|---:|---:|---:|---:|---:|
| exp1 | π | 0.15 | **0.7824** | 0.7297 | 0.7247 | +0.0294 |
| exp2 | π | 0.45 | 0.7794 | 0.6757 | 0.6754 | +0.0265 |
| exp3 | π | 0.75 | 0.7529 | 0.5405 | 0.4638 | 0.0000 |
| exp4 | π/2 | 0.15 | 0.7618 | 0.7027 | 0.7018 | +0.0088 |
| exp5 | π/5 | 0.15 | 0.7235 | 0.6216 | 0.6213 | −0.0294 |
| exp6 | π/10 | 0.15 | 0.7000 | 0.7027 | 0.6992 | −0.0529 |
| exp7 | π/10 | 0.45 | 0.7000 | 0.7027 | 0.7018 | −0.0529 |
| exp8 | π/10 | 0.75 | 0.7265 | 0.7297 | 0.7247 | −0.0265 |

**直接观察：**

- 最佳 AUROC 是 exp1（frequency、π、0.15），0.7824；比 raw 高 0.0294，但 Accuracy/F1 与 raw 相同，说明优势主要体现在排序能力而非固定阈值后的分类结果。
- frequency 组的 AUROC 随 coherence threshold 从 0.15 到 0.75 单调下降（0.7824 → 0.7794 → 0.7529）；Accuracy/F1 则在 exp3 明显恶化，说明过高门槛可能留下不均匀的伪迹残差或造成特征分布不稳定。
- phasegated 组中 exp4（π/2）最好；继续收紧到 π/5、π/10 后，AUROC 降至 0.7235、0.7000。就本批结果而言，更严格相位门控没有带来更好的泛化。
- 7 月 10 日的 raw/ICA 基线与所有归档中的数据规模一致：train/val/test subject 数为 124/17/37，test 为 20 个 epilepsy 与 17 个 control subject。

## 3. SHAP 分布总览

归档的 SHAP 聚合仍然显示相同的主轴：Hjorth 与 theta 在 8 组中始终排名前二；通道层面 T5 在 8 组中均为第一，随后常见的是 F3、FP1、T6、C4、Pz。因而加权融合没有把模型解释完全转移到某个重叠通道；模型仍主要利用左颞后链及背景形态/慢波特征。

| 实验 | mean(|SHAP|) 前三类 | 通道 Top 3 |
|---|---|---|
| exp1 | Hjorth 0.01876；theta 0.01483；gamma 0.00680 | T5 0.04090；F3 0.02101；T6 0.02060 |
| exp2 | Hjorth 0.02297；theta 0.01536；gamma 0.00782 | T5 0.04649；F3 0.02754；FP1 0.02714 |
| exp3 | Hjorth 0.00796；theta 0.00506；alpha 0.00344 | T5 0.02028；C4 0.01199；F3 0.01032 |
| exp4 | Hjorth 0.03482；theta 0.02625；gamma 0.01213 | T5 0.05019；FP1 0.04510；F8 0.03299 |
| exp5 | Hjorth 0.02867；theta 0.01881；gamma 0.00888 | T5 0.05201；FP1 0.03491；F3 0.02890 |
| exp6 | Hjorth 0.02275；theta 0.01687；gamma 0.00798 | T5 0.04727；FP1 0.02557；F3 0.02282 |
| exp7 | Hjorth 0.01984；theta 0.01067；gamma 0.00604 | T5 0.04411；F3 0.02315；FP1 0.01868 |
| exp8 | Hjorth 0.03285；theta 0.02000；gamma 0.01298 | T5 0.05375；FP1 0.03734；F3 0.03209 |

这里的 gamma 实际只有 30–40 Hz，受 `bandpass=[0.5,40]` 截断，不能解释为 HFO。SHAP 也不能单独证明被滤除的成分是伪迹还是神经耦合；它只描述分类器在当前特征共线结构下的归因分配。

## 4. 相位门控与生理解释

`phase=π` 等价于不做相位限制，回归掉任意相位的组内共享成分。exp1/exp2 的 AUROC 最高，说明在当前数据和 211 维特征空间中，较激进的共享成分去除并未明显破坏分类所需的颞区背景信息。T5、T6 和 theta/Hjorth 的稳定高贡献，仍支持“模型利用颞区背景形态与慢波组织性”的解释。

但这不等于证明 frequency 方法没有误删真实耦合：双侧枕区同步 PDR 或颞区传播活动都可能具有较高 coherence，单靠 211 维功率/Hjorth/谱熵特征无法识别其相位结构。相位门控收紧到 π/10 后性能下降，反而说明“更保守”不必然等于“更生理”；它可能保留了额部伪迹，也可能改变了真实背景信号与下游特征的相关结构。

## 5. 重叠通道加权融合是否有显著影响

### 5.1 本次实现改变了什么

旧实现对 `T3/O1/T4/O2` 采用按组列表顺序的后写覆盖：若两个组都通过门限，后遍历的组会覆盖前一组的结果。新实现先为每个目标通道收集来自多个组的 coherent 候选，再按目标-参考 coherence 归一化加权，输出 `channel_weights` 和 `channel_sources` 诊断信息。这样，结果不再取决于 `channel_groups` 的书写顺序。

### 5.2 同配置跨日期准对照

7 月 9 日归档对应旧覆盖逻辑，7 月 10 日归档对应加权融合。两批实验配置、subject split 和模型流程保持相同，因此可用来观察方向性变化；但由于没有保存逐 subject prediction，无法做 paired bootstrap、DeLong 或 permutation test。

| 实验 | 7/9 AUROC | 7/10 AUROC | ΔAUROC | 7/9 F1 | 7/10 F1 | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|
| exp1 | 0.7618 | 0.7824 | +0.0206 | 0.7018 | 0.7247 | +0.0229 |
| exp2 | 0.6912 | 0.7794 | +0.0882 | 0.5836 | 0.6754 | +0.0918 |
| exp3 | 0.7235 | 0.7529 | +0.0294 | 0.6992 | 0.4638 | −0.2354 |
| exp4 | 0.6794 | 0.7618 | +0.0824 | 0.6735 | 0.7018 | +0.0283 |
| exp5 | 0.6559 | 0.7235 | +0.0676 | 0.6146 | 0.6213 | +0.0068 |
| exp6 | 0.6971 | 0.7000 | +0.0029 | 0.6992 | 0.6992 | 0.0000 |
| exp7 | 0.7500 | 0.7000 | −0.0500 | 0.6696 | 0.7018 | +0.0322 |
| exp8 | 0.7088 | 0.7265 | +0.0176 | 0.6947 | 0.7247 | +0.0300 |

8 组中 7 组 AUROC 上升，AUROC 增量中位数约 +0.025；但 exp7 下降 0.0500，且 exp3 的 F1 下降 0.2354，说明影响并非稳定的单向增益。Accuracy 也没有一致改善。更合理的结论是：

> 加权融合很可能改变了重叠通道的信号与特征分布，并在多数配置下改善 AUROC；但现有归档不足以证明这种改善具有统计显著性，也不足以证明提升专门来自 T3/O1/T4/O2，而不是代码重跑、模型随机性或阈值变化。

### 5.3 为什么不能把结果写成“显著提升”

- test 只有 37 个 subject，点估计波动可能很大；
- 未归档旧/新版本的逐 subject prediction，不能进行配对检验；
- 未归档每个 epoch 的 `channel_weights`、`channel_sources` 和候选通过/跳过状态，无法确认四个重叠通道实际用了哪些组、权重是多少；
- 7 月 10 日同时修复了候选筛选、`skipped_pairs` 归档和若干 connectivity/verification 代码，跨日期差异不能严格归因于融合一项。

因此，本次数据对“后写覆盖可能引入顺序依赖”的工程性修复提供了支持，但对“加权融合带来显著临床/分类收益”尚无充分证据。至少从 SHAP 排名看，没有出现重叠通道突然成为唯一主导特征的现象，说明融合没有造成明显的解释崩塌。

## 6. 结论

- 本批最佳条件是 exp1：frequency、`coherence_threshold=0.15`，Test AUROC 0.7824；但 Accuracy/F1 未超过 raw。
- 更严格的相位门控在本批没有改善 test 泛化，π/10 组的 AUROC 均低于 raw。
- 加权融合后的 8 组结果整体较 7 月 9 日旧覆盖结果更好（AUROC 7/8 组上升），但效应方向和幅度不稳定，不能称为统计显著。
- 当前最稳妥的表述是：**加权融合消除了重叠通道的组顺序依赖，并可能带来中等的 AUROC 改善；其对真实神经耦合保护或临床性能的独立贡献仍待专门消融验证。**

## 7. 局限与下一步

1. 在同一固定 split 上运行真正的双版本消融：`coherence_weighted` vs `legacy_last_write`，只改变融合策略，保存逐 subject test prediction。
2. 在 `skipped_pairs_subject.csv` 之外归档每个 epoch 的 `channel_sources`、`channel_weights`、候选 coherence、phase-gate pass fraction，并按重叠通道统计有效融合比例和权重分布。
3. 对 AUROC、F1、Accuracy 做 subject-level paired bootstrap/DeLong 置信区间；报告效应量，而不是只比较点估计。
4. 归档 V1 coherence 前后变化，并启用 connectivity/wPLI/PLV 特征，直接检验加权融合和相位门控是否保留非零相位神经耦合。
5. 对 T3/O1/T4/O2 做单通道敏感性分析：分别只使用前组、后组、加权融合候选，观察性能和 SHAP 是否稳定。
