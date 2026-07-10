# Wiener phase/coherence grid experiment design

## Scope

Rerun the fixed eight-cell Wiener phase/coherence ablation after the target-level
coherence-weighted overlap fusion change. The only experimental variables remain
`phase_gate_threshold_rad` and `coherence_threshold`; `random_seed=42`, subject
splits, channel groups, preprocessing, and model configuration remain fixed.

The overlap policy is an invariant implementation detail:
`overlap_policy=coherence_weighted`. It is not a third experimental factor.

Each cell produces two paired downstream feature readouts:

- `base211`, the backward-compatible primary readout;
- `base211_conn80`, a secondary readout that appends connectivity features.

The feature profile is a fixed readout track, not an additional Wiener variable.

## Matrix

| Cell | Mode | Phase threshold | Coherence threshold |
|---|---|---:|---:|
| exp1 | frequency | π | 0.15 |
| exp2 | frequency | π | 0.45 |
| exp3 | frequency | π | 0.75 |
| exp4 | phasegated | π/2 | 0.15 |
| exp5 | phasegated | π/5 | 0.15 |
| exp6 | phasegated | π/10 | 0.15 |
| exp7 | phasegated | π/10 | 0.45 |
| exp8 | phasegated | π/10 | 0.75 |

## Data flow

1. Script 01 extracts the shared epoch cache once.
2. For each cell, script 02 rebuilds the mode-specific Wiener cache with target
   candidate diagnostics and fusion weights.
3. Script 04 reads that cache and writes V1 coherence, gate, skipped-pair,
   fusion, and connectivity summaries.
4. Script 06 trains both feature profiles into profile-isolated result trees.
5. Script 07 archives both profiles, verification summaries, resolved config, and
   prediction CSVs.
6. A grid analyzer compares all cells with subject-level paired bootstrap and
   FDR-adjusted physical verification summaries.

## Persisted fusion diagnostics

`WienerResult` exposes `candidate_fusion_weight`, aligned with `candidate_keys`.
Script 02 stores it as `(n_epochs, n_candidates)` and increments the cache schema
version. Rejected candidates have weight zero; accepted candidates for one output
channel sum to one. This permits compact aggregation of T3/O1/T4/O2 multi-source
fusion without archiving per-epoch signal arrays.

## Result layout

```text
results/<experiment>/xgboost/<feature_set>/<condition>/
results/<experiment>/verification/
experiments/<timestamp>_.../xgboost/<feature_set>/<condition>/
experiments/<timestamp>_.../verification/
```

Both XGBoost validation and test prediction CSVs are archived so downstream
comparisons can use the identical 37 test subjects.

## Statistical outputs

The grid analyzer reports:

- subject-level Test AUROC/F1/Accuracy per cell and feature profile;
- paired bootstrap confidence intervals against raw and between feature profiles;
- V1 coherence reduction by pair role and band;
- connectivity changes for coherence, PLV, imaginary coherence, and wPLI;
- gate acceptance, skipped groups, and overlap fusion rates/weights;
- Benjamini–Hochberg FDR q-values for exploratory multi-comparison summaries.

The analysis does not claim a causal effect of weighted fusion versus legacy
last-write behavior, because the new grid has weighted fusion fixed in every cell.

## Acceptance criteria

- All eight configs differ only in the two declared Wiener variables and result
  path.
- Training seed remains 42.
- `base211` and `base211_conn80` never overwrite one another.
- Every archived cell contains verification and prediction artifacts.
- Every overlapping channel has an auditable multi-source rate and weight summary.
- Old single-profile archives remain readable by the organizer.
