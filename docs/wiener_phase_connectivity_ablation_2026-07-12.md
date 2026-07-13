# Wiener 相位门控、coherence 阈值与 connectivity 特征实验总结

**数据源：** `experiments/2026-07-12_17*_exp_wiener_phase_*/`（17:11:43–17:12:23，exp1–exp8）  
**生成日期：** 2026-07-12  
**模型：** XGBoost；`base211` 与 `base211_conn80` 两条固定特征读出  
**指标口径：** test subject-level；测试集 37 人（epilepsy 20、control 17），训练/验证/测试 subject 数为 124/17/37  
**统计补充：** 基于归档的逐 subject prediction，使用分层 paired bootstrap（20,000 次，seed=20260712）估计 AUROC 差值的 95% CI

本报告沿用 `docs/wiener_phase_ablation_2026-07-10.md` 的结构，重点分析新增 80 维 connectivity 后，相位门控和 coherence 阈值的含义是否改变，以及分类结果能够支持到什么程度的降噪结论。

需要先说明一个关键限制：本次 8 个 `experiment.json` 均记录 `has_verification: false`，归档中没有 V1 coherence reduction、gate pass fraction、candidate skip/fusion、imaginary coherence 或 wPLI 汇总。因此，本报告可以直接比较**分类读出和 SHAP 归因**，但无法直接量化“去除了多少共同成分”或“保留了多少非零相位神经耦合”。下文凡涉及降噪强弱，均明确区分直接证据与机制推断。

## 1. 实验变量与 connectivity 读出

8 组 Wiener 参数矩阵与 7 月 10 日实验一致；`overlap_policy=coherence_weighted`、数据划分、通道组、预处理和随机种子均固定。

| 实验 | 方法 | condition | 相位阈值 | coherence threshold |
|---|---|---|---:|---:|
| exp1 | frequency | wiener | π | 0.15 |
| exp2 | frequency | wiener | π | 0.45 |
| exp3 | frequency | wiener | π | 0.75 |
| exp4 | phasegated | wiener_phasegated | π/2 | 0.15 |
| exp5 | phasegated | wiener_phasegated | π/5 | 0.15 |
| exp6 | phasegated | wiener_phasegated | π/10 | 0.15 |
| exp7 | phasegated | wiener_phasegated | π/10 | 0.45 |
| exp8 | phasegated | wiener_phasegated | π/10 | 0.75 |

`base211_conn80` 在原 211 维特征之后增加 80 维双侧 connectivity：8 个同源电极对 × 5 个频带 × coherence/PLV 两种指标。8 对为 FP1–FP2、F3–F4、F7–F8、C3–C4、T3–T4、T5–T6、P3–P4、O1–O2。

这 80 维是**下游观察量**，不参与 Wiener 滤波器估计或门控。Wiener 的通道组描述运动伪迹传导路径，而 connectivity 描述双侧同源耦合；因此它更适合检查滤波后的网络结构是否仍具有分类信息，不能被解释为额外的滤波约束。

## 2. Test 性能结果

### 2.1 Profile-specific baseline

| 特征集 | 条件 | Test AUROC | Test Accuracy | Test F1 |
|---|---|---:|---:|---:|
| base211 | raw | 0.7588 | 0.7027 | 0.6992 |
| base211 | ICA | 0.6500 | 0.6486 | 0.6210 |
| base211_conn80 | raw | **0.7853** | 0.7027 | 0.6793 |
| base211_conn80 | ICA | **0.7324** | 0.6757 | 0.6442 |

connectivity 对 raw 的 AUROC 增加 0.0265，但 F1 下降 0.0199；对 ICA 的 AUROC 增加 0.0824，同时 Accuracy/F1 小幅上升。由此可见，connectivity 确实提供了额外排序信息，但并不保证验证集阈值迁移到 test 后的分类指标同步改善。

### 2.2 Wiener 参数组：base211 与 base211_conn80 对照

| 实验 | 相位 | coh | base211 AUROC / Acc / F1 | conn80 AUROC / Acc / F1 | ΔAUROC（conn80−base211） |
|---|---|---:|---:|---:|---:|
| exp1 | π | 0.15 | 0.7529 / 0.7297 / 0.7279 | 0.7588 / 0.7297 / 0.7035 | +0.0059 |
| exp2 | π | 0.45 | 0.7794 / 0.7027 / 0.7018 | 0.7882 / 0.6486 / 0.6073 | +0.0088 |
| exp3 | π | 0.75 | 0.7588 / 0.5405 / 0.4638 | **0.8176 / 0.7838 / 0.7824** | +0.0588 |
| exp4 | π/2 | 0.15 | 0.7118 / 0.7027 / 0.7018 | 0.7147 / 0.5676 / 0.5673 | +0.0029 |
| exp5 | π/5 | 0.15 | 0.6794 / 0.5946 / 0.5946 | 0.7441 / 0.7568 / 0.7448 | +0.0647 |
| exp6 | π/10 | 0.15 | 0.7029 / 0.6216 / 0.6191 | 0.7794 / 0.6486 / 0.6476 | +0.0765 |
| exp7 | π/10 | 0.45 | 0.7471 / 0.6757 / 0.6754 | 0.6941 / 0.5946 / 0.5626 | −0.0529 |
| exp8 | π/10 | 0.75 | 0.7088 / 0.6757 / 0.6754 | **0.7912 / 0.8108 / 0.8057** | +0.0824 |

**直接观察：**

- `base211_conn80` 的最高 AUROC 是 exp3（π、0.75），0.8176；比同 profile 的 raw 高 0.0324。最高 Accuracy/F1 是 exp8（π/10、0.75），0.8108/0.8057，比 raw 分别高 0.1081/0.1264。
- exp8 的 validation AUROC 也最高（0.7778），exp6 次之（0.7639）；exp3 的 validation AUROC 为 0.6944。因此，exp3 是 test 排序最优点，exp8 则在 validation/test 一致性和阈值后指标上更稳健。
- connectivity 对 8 个 Wiener 单元并非一致增益：7 组 AUROC 上升、exp7 下降 0.0529；Accuracy/F1 在 exp1、exp2、exp4、exp7 中没有随 AUROC 同步改善。
- 逐 subject paired bootstrap 显示，8 组 `conn80 Wiener − conn80 raw` 的 AUROC 95% CI 均跨 0；`conn80 − base211` 的 AUROC 95% CI 也全部跨 0。37 人测试集下，这些差异应表述为趋势和候选参数，而非统计显著优势。

## 3. 加入 connectivity 后，参数趋势如何改变

### 3.1 无相位门控：coherence 阈值趋势发生反转

frequency 组在 `base211_conn80` 下随 coherence threshold 从 0.15 增至 0.75，AUROC 单调上升：

`0.7588 → 0.7882 → 0.8176`

相比之下，`base211` 是 `0.7529 → 0.7794 → 0.7588`，高阈值并没有优势。新增 connectivity 后，exp3 不仅恢复了排序能力，还将 Accuracy/F1 从 base211 的 0.5405/0.4638 提升到 0.7838/0.7824。

最合理的解释不是“0.75 去噪更多”，而是：较高的 candidate coherence 门槛减少了对弱或不稳定共享关系的处理，保留的双侧耦合结构被 connectivity 特征重新利用。原 211 维只观察单通道功率、Hjorth、谱熵和功率不对称，无法表达这种补偿信息，因此会低估高阈值方案的适用性。

### 3.2 固定低 coherence 阈值：收紧相位门控后逐步恢复

在 `coherence_threshold=0.15` 时，conn80 的 phasegated AUROC 为：

`π/2: 0.7147 → π/5: 0.7441 → π/10: 0.7794`

更严格的零相位门控逐步接近 raw conn80 的 0.7853。这说明低 coherence 阈值允许较多 candidate 进入后，必须依靠更窄的相位窗口限制真正被回归掉的频率系数；π/2 对当前数据过宽，可能同时改变了与疾病相关的耦合结构。

但 π/10、0.15 的 exp6 只是在 AUROC 上接近 raw，Accuracy/F1 仍较低。它更像“减少伤害”的保守设置，而不是已经证明优于不滤波的方案。

### 3.3 固定 π/10：coherence 阈值表现非单调

π/10 组的 conn80 AUROC 为：

`0.15: 0.7794 → 0.45: 0.6941 → 0.75: 0.7912`

中间阈值 exp7 明显最差，说明两个硬门控是交互关系，不能把 coherence threshold 简单理解为连续的“降噪旋钮”。在某些 epoch/通道上，0.45 可能改变 candidate 的通过组合和重叠通道融合来源，造成群体间不一致的特征分布；但由于本次没有归档 candidate/fusion 诊断，这一机制尚不能被验证。

## 4. 两个门控参数的准确含义与适用情况

### 4.1 coherence threshold 不是普通相关系数阈值

代码中的阈值作用于每个“目标通道—组内参考通道” candidate。分数是 0.5–40 Hz 内、所有参考通道和频点上的**最大 magnitude-squared coherence**。只有该最大值达到阈值，candidate 才进入 Wiener 估计；它不是 Pearson correlation、不是全频带平均 coherence，也不是最终去除比例。

因此：

- 0.15 较宽松，更多 candidate 有资格被处理，但不代表每个频点都会被强力回归；
- 0.75 较严格，倾向于只处理存在很强窄带共享成分的 candidate，更适合“避免误伤弱耦合”的保守策略；
- 因为使用频带内最大值，单个高 coherence 频点即可触发 candidate。阈值升高与实际处理率、移除能量不必线性对应，必须结合 candidate acceptance rate 和 removed-energy ratio 才能解释为降噪强度。

在当前 conn80 结果中，0.75 分别产生了 AUROC 最优 exp3 和 Accuracy/F1 最优 exp8，可作为下一轮验证的优先阈值；0.45 的表现高度依赖相位设置，不宜作为默认折中值。

### 4.2 phase gate 控制的是频率系数的零相位邻域

相位门控对已经通过 coherence eligibility 的 candidate 生效。对每个参考通道 × 频率 bin，只有交叉谱相位距离满足 `|phase| ≤ threshold`，对应 Wiener 系数才保留：

- π：全部相位通过，等价于普通 frequency Wiener；
- π/2：允许较宽的相位范围；
- π/5、π/10：逐步聚焦近零相位共享成分；被门控拒绝的成分不会进入 coherent estimate，而是保留在 specific 输出中。

严格门控的生理动机是优先去除近零时延的体积传导/共同伪迹，同时保留非零相位传播。但“近零相位”不等于“伪迹”：真实双侧同步背景活动也可能接近零相位；反之，运动伪迹也可能因传导路径和滤波出现相位差。因此，相位阈值只能作为结构先验，不能单独完成伪迹判别。

### 4.3 推荐按任务目标选择，而不是寻找单一全局最优

| 使用目标 | 当前优先候选 | 原因 | 仍需验证 |
|---|---|---|---|
| 以 AUROC/排序为主 | exp3：π、0.75 | test AUROC 0.8176，为全网格最高 | validation AUROC 仅 0.6944；需外部测试和校准 |
| 兼顾阈值后分类与跨 split 一致性 | exp8：π/10、0.75 | validation AUROC 0.7778；test Acc/F1 全网格最高 | AUROC 对 raw 仅 +0.0059，CI 跨 0 |
| 保守探索近零相位去噪 | exp6：π/10、0.15 | test AUROC 0.7794，接近 raw；相位限制最明确 | Acc/F1 较低，尚无直接降噪验证 |
| 当前不建议默认使用 | exp4、exp7 | conn80 AUROC 0.7147/0.6941，且阈值后指标较差 | 需检查 candidate 通过率和融合来源是否异常 |

如果只能选一个参数进入下一阶段直接降噪验证，建议优先 exp8；如果部署目标明确以排序为主，并允许独立校准分类阈值，可并行保留 exp3。

## 5. Connectivity SHAP 的信息

### 5.1 Connectivity 已成为可见的模型信息源

`base211_conn80` 中 raw 的 connectivity mean(|SHAP|) 为 0.01318。Wiener 各组为：

| 实验 | exp1 | exp2 | exp3 | exp4 | exp5 | exp6 | exp7 | exp8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| connectivity mean(|SHAP|) | 0.01522 | 0.02034 | 0.00793 | 0.00356 | 0.01953 | 0.01239 | 0.00099 | **0.03366** |

exp8 的 connectivity 归因最高，且仅次于 Hjorth 聚合；其 SHAP top-20 中出现 `plv_F3_F4_beta`、`coh_F7_F8_gamma`、`coh_FP1_FP2_gamma`、`coh_C3_C4_beta/theta`、`plv_C3_C4_beta` 和 `plv_F7_F8_theta`。exp3 中也出现 `plv_F3_F4_beta`、`coh_FP1_FP2_theta/beta`、`coh_F7_F8_delta` 和 `plv_F7_F8_alpha`。

这说明高阈值方案没有把下游判别完全压缩到单通道功率；双侧耦合仍可参与判别。尤其 exp8 在严格近零相位门控后仍有较高 connectivity SHAP，符合“保留了可辨识连接结构”的假设。

### 5.2 不能把较大 SHAP 直接等同于更好的连接保留

SHAP 大小受树模型选择、特征尺度、共线性和整体模型输出幅度影响。exp8 的所有聚合 SHAP 都偏大，connectivity=0.03366 不能单独证明其物理 connectivity 比其他组保留更多。相反，exp3 的 connectivity SHAP 只有 0.00793，却取得最高 AUROC，说明性能并不由 connectivity 归因强度单调决定。

exp7 的 connectivity SHAP 几乎坍缩到 0.00099，并伴随最低 conn80 AUROC；这是值得优先排查的异常点，但仍需原始特征分布和 candidate 诊断确认原因。

## 6. 由此引出的降噪性能分析

### 6.1 本次数据直接支持的结论

- connectivity 改变了参数排序：无相位门控时，高 coherence threshold 从 base211 的普通表现变为 conn80 下的最优 AUROC。
- 最好的 conn80 方案集中在 `coherence_threshold=0.75`，提示下游模型更受益于选择性处理强共享成分，而不是广泛处理弱共享关系。
- π/10、0.75 的 exp8 同时具有最佳 validation AUROC 和 test Accuracy/F1，并保留较强的 connectivity 模型贡献，是当前最有希望平衡伪迹去除与连接保留的设置。
- 低阈值 + 宽相位门控（exp4）和中阈值 + 严格相位门控（exp7）表现较差，说明“更宽松”或“更严格”都不能单独预测结果；candidate eligibility、频率相位门控和重叠通道融合共同决定输出。

### 6.2 不能由本次数据直接支持的结论

- 不能声称 exp3 或 exp8 去除了最多伪迹；分类性能高可能来自较少处理、特征补偿或模型校准，而非更强降噪。
- 不能声称 exp8 保留了更多真实神经连接；现有 connectivity 是滤波后信号的分类特征，没有 raw→specific 的成对物理变化统计。
- 不能把 exp7 的下降归因于 coherence=0.45 本身；缺少 candidate acceptance、phase pass fraction、fusion weight 和每个通道的 removed-energy 数据。
- 不能声称任一方案显著优于 raw；paired bootstrap AUROC CI 全部跨 0，且测试集只有 37 人。

### 6.3 当前最符合数据的工作假设

> 新增 connectivity 后，模型更能利用 Wiener 输出中残留或保留的双侧耦合拓扑。当前数据更支持“选择性、保守地处理高可信共享成分”，而不是“尽可能扩大被处理的 candidate 范围”。exp8 是平衡阈值后性能与连接信息的首选候选，exp3 是排序性能候选；两者都必须通过直接信号域验证后才能称为降噪最优。

## 7. 统计不确定性

下面列出 `base211_conn80` 下 Wiener 相对同 profile raw 的 paired bootstrap AUROC 差值：

| 实验 | ΔAUROC vs raw | 95% CI |
|---|---:|---:|
| exp1 | −0.0265 | [−0.1471, 0.0824] |
| exp2 | +0.0029 | [−0.1176, 0.1059] |
| exp3 | +0.0324 | [−0.0794, 0.1412] |
| exp4 | −0.0706 | [−0.2147, 0.0618] |
| exp5 | −0.0412 | [−0.1676, 0.0706] |
| exp6 | −0.0059 | [−0.1412, 0.1176] |
| exp7 | −0.0912 | [−0.2559, 0.0706] |
| exp8 | +0.0059 | [−0.1088, 0.1206] |

exp3 的点估计最高、exp8 的阈值后指标最好，但置信区间较宽。这里还同时比较了 8 个网格点和两个 profile，未做独立外部确认；参数选择应以 validation、机制诊断和复现实验共同决定，不能按 test 最大值单独定稿。

## 8. 结论

- 新增 connectivity 后，参数选择结论发生实质变化：frequency 模式从低阈值偏好转为 `coherence_threshold=0.75` 的 exp3 获得最高 AUROC。
- 对严格相位门控，coherence threshold 呈明显非单调；0.45 的 exp7 最差，0.75 的 exp8 最好，证明两个门控必须作为二维交互参数解释。
- 当前首选是 exp8（π/10、0.75）：validation/test 更一致，test Accuracy/F1 为 0.8108/0.8057，connectivity 也成为重要模型信息源。exp3（π、0.75）作为 AUROC 优先候选并行保留。
- connectivity 的增益在 8 组中不稳定，所有 paired-bootstrap AUROC CI 均跨 0；现阶段应称为候选趋势，而非显著改善。
- 本批归档没有 verification 数据，因此最稳妥的总结是：**高 coherence 门槛下的选择性处理更适合 connectivity 读出，但其真实伪迹抑制、神经耦合保护和移除能量仍未被本次结果直接证明。**

## 9. 下一步

1. 重新归档 8 个单元的 V1、gate、skip、fusion 和 connectivity verification，确保 `has_verification=true`；按 subject 报告 raw→specific 的 coherence、PLV、imaginary coherence、wPLI 变化。
2. 增加 removed-energy ratio、PSD distortion、波形重构误差，并分别统计 Wiener 目标通道与 9 个 passthrough 通道，避免用分类性能替代降噪指标。
3. 对 exp3、exp8、raw 做同一 37 人的 paired bootstrap/置换检验，并在独立 split 或外部队列复现；阈值后指标应重新校准，而不是复用本次 test 观察。
4. 对 exp7 做诊断性复跑：比较 candidate acceptance、phase pass fraction、T3/O1/T4/O2 的多源融合率与权重，确认中间 coherence 阈值是否造成不均匀处理。
5. 将 connectivity 变化按“Wiener 组内同源对”（FP1–FP2、O1–O2）和“非直接组内双侧对”分层。若只在直接处理的同源对上大幅下降，更可能是滤波改写；若非直接组内对仍稳定保留，更支持网络结构未被普遍破坏。
