#!/usr/bin/env python3
"""Analyze existing ERP-CORE benchmark outputs without reloading EEG data.

This script only consumes ``subject_metrics.csv`` and, when present,
``run_summary.json``.  It performs participant-paired tests, participant
bootstrap confidence intervals, and TOST equivalence tests for ERP morphology.
It deliberately does not claim to compute trial-level SME or synthetic-signal
recovery because those require data that script 10 does not currently save.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRIC_DIRECTIONS = {
    "ern_snr_db": "higher",
    "ern_waveform_r": "higher",
    "ern_rmse_vs_standard_uv": "lower",
    "ern_peak_uv": "descriptive",
    "ern_peak_latency_ms": "descriptive",
    "baseline_noise_sd_uv": "lower",
    "lrp_peak_uv": "descriptive",
    "lrp_peak_latency_ms": "descriptive",
    "lrp_half_peak_onset_ms": "descriptive",
    "line_frequency_power_v2_hz": "lower",
    "fp1_fp2_proxy_variance_uv2": "lower",
    "classification_accuracy": "higher",
    "classification_f1": "higher",
    "classification_auc": "higher",
    "target_change_rms_uv": "descriptive",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对脚本 10 的现有被试级结果执行配对统计与等效性检验"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "erp_core_flankers",
        help="包含 subject_metrics.csv 的脚本 10 输出目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认写入 RESULTS_DIR/statistics",
    )
    parser.add_argument("--candidate", default="wiener", help="待评价方法")
    parser.add_argument(
        "--references",
        default="standard,raw",
        help="逗号分隔的参照方法；默认 standard,raw",
    )
    parser.add_argument(
        "--equivalence-reference",
        default="standard",
        help="ERP 等效性检验的参照方法",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--ern-amplitude-margin-uv",
        type=float,
        default=1.0,
        help="ERN 峰值幅度 TOST 等效界值（±µV，需在研究方案中预注册）",
    )
    parser.add_argument(
        "--ern-latency-margin-ms",
        type=float,
        default=10.0,
        help="ERN 峰值潜伏期 TOST 等效界值（±ms）",
    )
    parser.add_argument(
        "--lrp-amplitude-margin-uv",
        type=float,
        default=0.5,
        help="LRP 峰值幅度 TOST 等效界值（±µV）",
    )
    parser.add_argument(
        "--lrp-latency-margin-ms",
        type=float,
        default=10.0,
        help="LRP 峰值/起始潜伏期 TOST 等效界值（±ms）",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.bootstrap_repeats < 100:
        raise ValueError("--bootstrap-repeats 必须 >= 100")
    if not 0.0 < args.alpha < 0.5:
        raise ValueError("--alpha 必须位于 (0, 0.5)")
    for name in (
        "ern_amplitude_margin_uv",
        "ern_latency_margin_ms",
        "lrp_amplitude_margin_uv",
        "lrp_latency_margin_ms",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须 > 0")


def load_subject_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到被试级指标文件: {path}")
    frame = pd.read_csv(path)
    required = {"subject_id", "method", "ern_snr_db"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"subject_metrics.csv 缺少列: {missing}")
    duplicated = frame.duplicated(["subject_id", "method"], keep=False)
    if duplicated.any():
        rows = frame.loc[duplicated, ["subject_id", "method"]].to_dict("records")
        raise ValueError(f"每个被试/方法必须只有一行，发现重复: {rows[:5]}")
    return frame


def paired_values(
    frame: pd.DataFrame, metric: str, candidate: str, reference: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if metric not in frame.columns:
        return np.array([]), np.array([]), []
    subset = frame.loc[
        frame["method"].isin([candidate, reference]),
        ["subject_id", "method", metric],
    ]
    pivot = subset.pivot(index="subject_id", columns="method", values=metric)
    if candidate not in pivot or reference not in pivot:
        return np.array([]), np.array([]), []
    pivot = pivot[[reference, candidate]].dropna()
    return (
        pivot[candidate].to_numpy(dtype=float),
        pivot[reference].to_numpy(dtype=float),
        pivot.index.astype(str).tolist(),
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    repeats: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    draws = rng.choice(values, size=(repeats, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def _two_sided_p(values: np.ndarray) -> float:
    if values.size < 2:
        return np.nan
    if np.allclose(values, values[0]):
        return 1.0 if np.isclose(values[0], 0.0) else 0.0
    return float(stats.ttest_1samp(values, 0.0).pvalue)


def _wilcoxon_p(values: np.ndarray) -> float:
    if values.size < 2:
        return np.nan
    if np.allclose(values, 0.0):
        return 1.0
    return float(stats.wilcoxon(values, alternative="two-sided").pvalue)


def _bh_qvalues(pvalues: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.dropna().astype(float)
    if valid.empty:
        return result
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result.loc[valid.index] = restored
    return result


def paired_test(
    frame: pd.DataFrame,
    metric: str,
    candidate: str,
    reference: str,
    repeats: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    candidate_values, reference_values, _ = paired_values(
        frame, metric, candidate, reference
    )
    if candidate_values.size == 0:
        return None
    differences = candidate_values - reference_values
    ci_low, ci_high = bootstrap_mean_ci(differences, repeats, alpha, rng)
    sd = float(np.std(differences, ddof=1)) if differences.size >= 2 else np.nan
    direction = METRIC_DIRECTIONS.get(metric, "descriptive")
    oriented = differences if direction == "higher" else -differences
    if direction == "descriptive":
        n_better = np.nan
        n_non_tie = np.nan
        better_fraction = np.nan
        better_p = np.nan
    else:
        non_tie = ~np.isclose(oriented, 0.0)
        n_non_tie = int(non_tie.sum())
        n_better = int((oriented[non_tie] > 0).sum())
        better_fraction = n_better / n_non_tie if n_non_tie else np.nan
        better_p = (
            float(stats.binomtest(n_better, n_non_tie, 0.5).pvalue)
            if n_non_tie
            else np.nan
        )
    return {
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "direction": direction,
        "n_pairs": int(differences.size),
        "candidate_mean": float(np.mean(candidate_values)),
        "reference_mean": float(np.mean(reference_values)),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "sd_difference": sd,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "paired_t_p": _two_sided_p(differences),
        "wilcoxon_p": _wilcoxon_p(differences),
        "cohen_dz": float(np.mean(differences) / sd) if sd > 0 else np.nan,
        "n_better": n_better,
        "n_non_tie": n_non_tie,
        "better_fraction": better_fraction,
        "better_binomial_p": better_p,
    }


def tost_equivalence(
    differences: np.ndarray,
    margin: float,
    alpha: float,
) -> dict[str, Any]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    n = int(differences.size)
    if n < 2:
        return {
            "n_pairs": n,
            "mean_difference": float(np.mean(differences)) if n else np.nan,
            "margin": margin,
            "p_lower": np.nan,
            "p_upper": np.nan,
            "tost_p": np.nan,
            "ci_90_low": np.nan,
            "ci_90_high": np.nan,
            "equivalent": False,
        }
    mean = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1))
    if np.isclose(sd, 0.0):
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
        ci_low = ci_high = mean
    else:
        se = sd / np.sqrt(n)
        df = n - 1
        t_lower = (mean + margin) / se
        t_upper = (mean - margin) / se
        p_lower = float(stats.t.sf(t_lower, df))
        p_upper = float(stats.t.cdf(t_upper, df))
        critical = float(stats.t.ppf(1.0 - alpha, df))
        ci_low = mean - critical * se
        ci_high = mean + critical * se
    tost_p = max(p_lower, p_upper)
    return {
        "n_pairs": n,
        "mean_difference": mean,
        "margin": float(margin),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "tost_p": tost_p,
        "ci_90_low": ci_low,
        "ci_90_high": ci_high,
        "equivalent": bool(tost_p < alpha),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def _write_report(
    path: Path,
    paired: pd.DataFrame,
    equivalence: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    verdict = summary["verdict"]
    lines = [
        "# ERP 降噪统计检验（现有结果）",
        "",
        "本报告只使用脚本 10 已保存的被试级指标，不重新读取 EEG 原始数据。",
        "",
        "## 核心结论",
        "",
        f"- Wiener 相对 ICA 的 ERN SNR 均值差 95% 被试 bootstrap CI 是否完全高于 0：**{verdict['snr_increase_vs_equivalence_reference']}**。",
        f"- ERN 峰值幅度是否通过 TOST 等效性检验：**{verdict['ern_amplitude_equivalent']}**。",
        f"- ERN 峰值潜伏期是否通过 TOST 等效性检验：**{verdict['ern_latency_equivalent']}**。",
        f"- 同时满足 SNR 增加、幅度等效和潜伏期等效：**{verdict['supports_better_ern_exposure']}**。",
        "",
        "这里的 `False` 表示现有数据和预设界值不足以支持该结论，不等同于证明两方法没有差异。",
        "",
        "## SNR 配对结果",
        "",
        "| 候选 | 参照 | n | 均值差 (dB) | 95% bootstrap CI | 配对 t p | Wilcoxon p | 改善比例 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    snr_rows = paired.loc[paired["metric"] == "ern_snr_db"]
    for _, row in snr_rows.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['reference']} | {int(row['n_pairs'])} | "
            f"{row['mean_difference']:.3f} | [{row['bootstrap_ci_low']:.3f}, "
            f"{row['bootstrap_ci_high']:.3f}] | {row['paired_t_p']:.4g} | "
            f"{row['wilcoxon_p']:.4g} | {row['better_fraction']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## ERP/LRP 等效性检验",
            "",
            "TOST 的界值不是通用常数；应由生理意义、测量可靠性或预注册方案确定。本次运行使用命令行给定的界值。",
            "",
            "| 指标 | n | 候选−参照均值 | 等效界值 | 90% CI | TOST p | 等效 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in equivalence.iterrows():
        lines.append(
            f"| {row['metric']} | {int(row['n_pairs'])} | {row['mean_difference']:.3f} | "
            f"±{row['margin']:.3f} | [{row['ci_90_low']:.3f}, {row['ci_90_high']:.3f}] | "
            f"{row['tost_p']:.4g} | {bool(row['equivalent'])} |"
        )
    lines.extend(
        [
            "",
            "## 当前产物不能计算的检验",
            "",
            "- 单试次解析或 bootstrap SME，以及由 SME 定义的 SNR 下界；当前结果只保存被试级汇总指标，没有保存每个事件的波形或单试次振幅。",
            "- ERP 峰值振幅/潜伏期的单试次或嵌套 bootstrap 稳定性；需要逐试次 epoch。",
            "- 注入已知 ERP 真值后的恢复误差、幅度偏差与潜伏期偏差；需要重新处理信号或合成数据。",
            "",
            "`paired_tests.csv` 中同时提供配对 t、Wilcoxon、被试 bootstrap CI、效应量和被试改善比例；BH q 值只用于多指标探索性校正。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Path]:
    _validate_args(args)
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_dir / "statistics"
    )
    frame = load_subject_metrics(results_dir / "subject_metrics.csv")
    methods = set(frame["method"].astype(str))
    if args.candidate not in methods:
        raise ValueError(f"候选方法 {args.candidate!r} 不在数据中: {sorted(methods)}")
    references = [item.strip() for item in args.references.split(",") if item.strip()]
    missing_references = sorted(set(references) - methods)
    if missing_references:
        raise ValueError(f"参照方法不在数据中: {missing_references}")
    if args.equivalence_reference not in methods:
        raise ValueError(f"等效性参照方法不在数据中: {args.equivalence_reference}")

    rng = np.random.default_rng(args.bootstrap_seed)
    paired_rows: list[dict[str, Any]] = []
    for reference in references:
        if reference == args.candidate:
            continue
        for metric in METRIC_DIRECTIONS:
            row = paired_test(
                frame,
                metric,
                args.candidate,
                reference,
                args.bootstrap_repeats,
                args.alpha,
                rng,
            )
            if row is not None:
                paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    if paired.empty:
        raise ValueError("没有可用于配对分析的指标")
    paired["paired_t_q_bh"] = _bh_qvalues(paired["paired_t_p"])
    paired["wilcoxon_q_bh"] = _bh_qvalues(paired["wilcoxon_p"])

    equivalence_specs = [
        ("ern_peak_uv", args.ern_amplitude_margin_uv),
        ("ern_peak_latency_ms", args.ern_latency_margin_ms),
        ("lrp_peak_uv", args.lrp_amplitude_margin_uv),
        ("lrp_peak_latency_ms", args.lrp_latency_margin_ms),
        ("lrp_half_peak_onset_ms", args.lrp_latency_margin_ms),
    ]
    equivalence_rows: list[dict[str, Any]] = []
    for metric, margin in equivalence_specs:
        candidate_values, reference_values, _ = paired_values(
            frame, metric, args.candidate, args.equivalence_reference
        )
        result = tost_equivalence(candidate_values - reference_values, margin, args.alpha)
        result.update(
            {
                "candidate": args.candidate,
                "reference": args.equivalence_reference,
                "metric": metric,
            }
        )
        equivalence_rows.append(result)
    equivalence = pd.DataFrame(equivalence_rows)

    snr_match = paired.loc[
        (paired["metric"] == "ern_snr_db")
        & (paired["reference"] == args.equivalence_reference)
    ]
    snr_increase = bool(
        not snr_match.empty and float(snr_match.iloc[0]["bootstrap_ci_low"]) > 0.0
    )

    def equivalent(metric: str) -> bool:
        match = equivalence.loc[equivalence["metric"] == metric, "equivalent"]
        return bool(not match.empty and bool(match.iloc[0]))

    verdict = {
        "snr_increase_vs_equivalence_reference": snr_increase,
        "ern_amplitude_equivalent": equivalent("ern_peak_uv"),
        "ern_latency_equivalent": equivalent("ern_peak_latency_ms"),
    }
    verdict["supports_better_ern_exposure"] = bool(
        verdict["snr_increase_vs_equivalence_reference"]
        and verdict["ern_amplitude_equivalent"]
        and verdict["ern_latency_equivalent"]
    )

    run_summary_path = results_dir / "run_summary.json"
    run_summary = (
        json.loads(run_summary_path.read_text(encoding="utf-8"))
        if run_summary_path.exists()
        else None
    )
    summary = {
        "source_results_dir": str(results_dir),
        "candidate": args.candidate,
        "references": references,
        "equivalence_reference": args.equivalence_reference,
        "n_subject_rows": int(len(frame)),
        "n_unique_subjects": int(frame["subject_id"].nunique()),
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.bootstrap_seed,
        "alpha": args.alpha,
        "equivalence_margins": {
            "ern_amplitude_uv": args.ern_amplitude_margin_uv,
            "ern_latency_ms": args.ern_latency_margin_ms,
            "lrp_amplitude_uv": args.lrp_amplitude_margin_uv,
            "lrp_latency_ms": args.lrp_latency_margin_ms,
        },
        "verdict": verdict,
        "unavailable_without_trial_waveforms": [
            "analytic_or_bootstrap_SME",
            "SME_based_SNR_lower_bound",
            "trial_or_nested_bootstrap_peak_stability",
            "synthetic_known_truth_recovery",
        ],
        "source_run_summary": run_summary,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "paired_tests": output_dir / "paired_tests.csv",
        "equivalence_tests": output_dir / "equivalence_tests.csv",
        "summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
    }
    paired.to_csv(paths["paired_tests"], index=False)
    equivalence.to_csv(paths["equivalence_tests"], index=False)
    paths["summary"].write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(paths["report"], paired, equivalence, summary)
    return paths


def main() -> None:
    args = _parse_args()
    paths = analyze(args)
    print("ERP 统计分析完成：")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
