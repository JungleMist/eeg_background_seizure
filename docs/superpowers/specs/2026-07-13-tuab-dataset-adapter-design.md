# TUAB Dataset Adapter Design

Date: 2026-07-13

## Goal

Extend the existing TUEP-oriented preprocessing and XGBoost pipeline so it can
run on TUAB v3.0.1 without duplicating the pipeline or changing the established
TUEP behavior.

TUAB is a recording-level normal/abnormal classification corpus. Training stays
at epoch level, while epoch probabilities are averaged within each recording.
The record-level decision threshold is selected on a validation subset derived
from the official TUAB training partition and then held fixed for the official
evaluation partition.

CNN support for TUAB is explicitly outside this change.

## Confirmed Dataset Semantics

TUAB v3.0.1 uses this structure:

```text
edf/{train,eval}/{abnormal,normal}/01_tcp_ar/*.edf
```

A filename such as `aaaaamye_s001_t000.edf` contains:

- patient ID: `aaaaamye`
- session ID: `s001`
- token ID: `t000`
- recording ID: `aaaaamye_s001_t000`

The normal/abnormal label belongs to the selected recording from an EEG
session, not permanently to the patient. A patient may therefore have both
normal and abnormal recordings in the TUAB training partition. Those
recordings remain separate labeled examples and must never be averaged
together.

The official `train` and `eval` partitions are patient-disjoint. The official
`eval` partition is the pipeline test split. A validation split is selected
from the official `train` partition while keeping every patient's recordings
together.

## Prediction Protocol

Both datasets retain a label-1 probability with a common interpretation:

| Dataset | Label 0 | Label 1 | Aggregation unit |
|---|---|---|---|
| TUEP | epilepsy | control | patient |
| TUAB | abnormal | normal | recording |

XGBoost is trained and predicts at epoch level. For TUAB, the pipeline averages
`P(normal)` across all valid epochs belonging to one `recording_id`. The
macro-F1-optimal threshold is selected from validation recordings. A test
recording is labeled normal when its mean probability is at least that fixed
threshold and abnormal otherwise.

## Configuration

`dataset.active` selects a dataset adapter. The original TUEP dataset
configuration moves under `dataset.tuep`, and TUAB receives a parallel block.

```yaml
dataset:
  active: "tuep"  # tuep | tuab

  tuep:
    reference_scheme: "ar"
    montage_dir: "01_tcp_ar"
    classes:
      epilepsy:
        folder: "00_epilepsy"
        label: 0
      control:
        folder: "01_no_epilepsy"
        label: 1

  tuab:
    edf_dir: "edf"
    reference_scheme: "ar"
    montage_dir: "01_tcp_ar"
    train_partition: "train"
    eval_partition: "eval"
    validation_fraction: 0.10
    max_recording_sec: 1200.0
    classes:
      abnormal:
        folder: "abnormal"
        label: 0
      normal:
        folder: "normal"
        label: 1
```

`paths.data_root` continues to identify the root of the active dataset. TUAB
uses a distinct `cache_dir` and `results_dir` so its products cannot mix with
TUEP products.

`configs/default.yaml` remains TUEP by default. A new `configs/tuab.yaml`
inherits the default config, selects TUAB, and supplies TUAB-specific data,
cache, and result paths.

Configuration validation requires:

- `dataset.active` is `tuep` or `tuab`;
- the active dataset block contains all required keys;
- its class labels are distinct and exactly `{0, 1}`;
- reference scheme and montage directory agree;
- TUAB validation fraction is greater than zero and at most one half;
- TUAB maximum recording duration is at least one epoch;
- the configured data root exists before discovery begins.

## Dataset Adapter Boundary

`eeg_bg/io/dataset.py` exposes dataset-neutral entry points:

```python
build_recording_index(cfg) -> pandas.DataFrame
assign_dataset_splits(index, cfg) -> pandas.DataFrame
get_recording_intervals(row, cfg, duration) -> list[tuple[float, float]]
```

These functions dispatch using `dataset.active`. Dataset-specific directory
walking, ID parsing, split assignment, and interval selection stay in private
TUEP and TUAB helpers. `scripts/01_extract_epochs.py` consumes only the neutral
interface.

`build_subject_index()` remains as a compatibility wrapper for existing callers
while new pipeline code uses `build_recording_index()`.

## Canonical Recording Index

Both adapters produce the following columns:

```text
dataset_name
patient_id
session_id
token_id
recording_id
evaluation_id
class_name
label
reference
source_partition
edf_path
split
```

The identity rules are:

- TUEP `evaluation_id` remains the current label-prefixed patient ID, such as
  `00_patient123`, preserving cache and output compatibility.
- TUAB `evaluation_id` is its complete EDF stem and therefore equals
  `recording_id`.
- `patient_id` is always the raw patient identifier without a class prefix.
- `recording_id` always identifies one source EDF recording.

The canonical index is written to `cache/epochs/index.csv` after split
assignment and validation.

## TUAB Split Assignment

All records under official `eval` are assigned to `test` unchanged. Official
`train` records are divided into pipeline `train` and `val` splits with
`StratifiedGroupKFold`:

- rows are recording-level examples;
- `y` is the recording label;
- `groups` is `patient_id`;
- `n_splits` is `max(2, round(1 / validation_fraction))`, which is 10 for
  the confirmed 10% validation fraction;
- shuffling uses `split.random_seed`.

Every generated fold is scored by its deviation from the target validation
record count and target per-class recording counts. The lowest-scoring fold is
selected as validation; a deterministic fold-order tie break is used. This
supports patients with both labels because the group, rather than a derived
patient label, is the indivisible assignment unit.

Post-assignment checks require:

- no patient appears in more than one of train, validation, and test;
- every recording has exactly one label and one split;
- official eval records are test records;
- full training runs have both classes in train and validation.

An incomplete local download may contain only part of official eval. Script 01
may still index and extract those files, but it reports missing splits/classes.
Training scripts reject incomplete train/validation/test inputs before fitting.

## Epoch Selection

TUAB does not use TUEP `.csv_bi` annotations. Its usable interval is:

```python
stop_sec = min(recording_duration, cfg["dataset"]["tuab"]["max_recording_sec"])
intervals = [(0.0, stop_sec)]
```

With the confirmed defaults, every recording contributes at most the first 20
minutes. Existing non-overlapping 20-second slicing produces at most 60 epochs.
Recordings shorter than 20 minutes use their available duration, and a trailing
fragment shorter than one epoch is discarded. Existing amplitude-based artifact
rejection remains active. A recording with no valid epochs is skipped and
reported.

TUEP keeps its existing full-recording-minus-seizure-buffer interval behavior.

## Channel Handling

After channel normalization, channels are explicitly reordered to
`channels.standard_19`. TUAB recordings must contain every standard channel.
A TUAB recording missing one or more required channels is skipped with a clear
diagnostic rather than producing a differently shaped signal tensor. Extra EDF
channels remain ignored. Filtering and resampling behavior is otherwise
unchanged.

## Cache Identity and Metadata

The epoch cache key must change whenever epoch content can change. Its canonical
fingerprint covers:

```text
edf_path
dataset.active
target_sfreq
bandpass
channels.standard_19
epoch_length_sec
artifact_threshold_uv
TUEP seizure_buffer_sec or TUAB max_recording_sec
```

Each epoch cache stores:

```text
epochs
ch_names
dataset_name
patient_id
recording_id
evaluation_id
subject_id
class_name
label
split
source_partition
n_epochs
```

`subject_id` is a compatibility alias whose value equals `evaluation_id`.

`eeg_bg/io/cache.py` defines the canonical metadata keys and a helper that
copies present metadata from an input cache. Scripts 02 and 03 use that helper
when writing Wiener and ICA outputs rather than maintaining independent lists
of metadata fields. Scripts 04 and 05 prefer `evaluation_id` and fall back to
`subject_id` for legacy caches.

## XGBoost Data Flow

An internal feature dataset container carries identity metadata without
expanding positional tuples:

```python
@dataclass
class FeatureDataset:
    X: numpy.ndarray
    y: numpy.ndarray
    evaluation_ids: list[str]
    patient_ids: list[str]
```

New pipeline code uses `build_feature_dataset()`. Existing public
`build_dataset()` retains its three-value return shape for compatibility.
Feature caches add `evaluation_ids`, `patient_ids`, `dataset_name`, and the
existing schema hash.

XGBoost still fits on epoch rows. Its internal grid search changes from
epoch-random `StratifiedKFold` to patient-grouped `StratifiedGroupKFold`, with
patient IDs supplied through `GridSearchCV.fit(..., groups=patient_ids)`. The
final estimator refits on all train epochs and uses the independent validation
set for early stopping.

The shared aggregation function validates that each `evaluation_id` has one
true label, then produces an arithmetic mean of its epoch probabilities. It
returns evaluation ID, patient ID, epoch count, mean label-1 probability, and
true label. The validation-derived threshold adds the predicted label.

## Prediction and Metrics Outputs

XGBoost validation and test prediction CSVs contain:

```text
evaluation_id
subject_id
patient_id
recording_id
n_epochs
pred_proba
predicted_label
true_label
```

The compatibility `subject_id` column equals `evaluation_id`. Existing result
consumers that select `subject_id`, `pred_proba`, and `true_label` continue to
work even when additional columns are present.

Metrics JSON records include:

```text
positive_class
aggregation_unit
threshold
auroc
f1
accuracy
```

TUAB writes `positive_class=normal` and `aggregation_unit=recording`; TUEP
writes `positive_class=control` and `aggregation_unit=subject`.

In prediction CSVs, `recording_id` equals `evaluation_id` for TUAB. It is left
empty for TUEP because one patient-level prediction may aggregate epochs from
multiple source recordings.

Data statistics become class-name aware and separately count patients,
evaluation units, and epochs. The legacy `n_subjects` key remains as an alias
for `n_evaluation_units` during the compatibility period.

## CNN Scope

TUAB CNN support is not implemented. No TUAB-specific changes are made to
`eeg_bg/ml/cnn_dataset.py`, `eeg_bg/ml/cnn_pipeline.py`, or EEGNet.

`scripts/08_train_cnn.py` fails immediately with a clear unsupported-dataset
message when `dataset.active` is TUAB. `README.md` explicitly documents that
TUAB currently supports scripts 01 through 07, while CNN remains TUEP-only.

## Failure Behavior

Script 01 fails before worker creation when the data root is missing or no EDF
files are found. A corrupt EDF, missing required TUAB channel, or recording with
no valid epoch is a per-recording skip. Final output reports completed, cached,
and skipped recording counts plus split/class recording and epoch totals.

Script 06 fails before training when:

- train, validation, or test is empty;
- train or validation lacks either class;
- a patient crosses dataset splits;
- an evaluation unit has conflicting labels;
- a TUAB legacy feature cache lacks patient identity and cannot support grouped
  cross-validation.

The last case instructs the user to rerun script 06 with `--force` after
rebuilding the epoch caches.

## Testing

Unit and integration coverage includes:

1. new configuration structure, active dataset validation, and inheritance;
2. unchanged TUEP discovery, labels, and subject-level split behavior;
3. TUAB patient/session/token/recording parsing;
4. official eval-to-test mapping;
5. mixed-label patients remaining wholly in train or validation;
6. deterministic validation selection and approximate class preservation;
7. TUAB interval truncation to `min(duration, 1200)`;
8. 60 epochs from a complete 20-minute interval and correct short-recording
   tail handling;
9. cache fingerprint sensitivity to every content-changing setting;
10. metadata propagation through Wiener and ICA caches;
11. recording-level probability averaging and conflicting-label rejection;
12. validation-only threshold selection;
13. patient-grouped XGBoost cross-validation;
14. TUAB fail-fast behavior in script 08;
15. an optional integration test against the local TUAB sample, marked
    `integration`.

Primary verification is:

```bash
conda run -n eeg_pipeline python -m pytest tests/ -m "not integration"
```

The local-data integration check is run separately when the configured TUAB
sample is present.

## Expected File Scope

Implementation is expected to touch:

```text
configs/default.yaml
configs/tuab.yaml
eeg_bg/config/settings.py
eeg_bg/io/dataset.py
eeg_bg/io/cache.py
eeg_bg/io/edf_reader.py
eeg_bg/features/extraction.py
eeg_bg/ml/xgb_pipeline.py
scripts/01_extract_epochs.py
scripts/02_run_wiener.py
scripts/03_run_ica.py
scripts/04_run_verification.py
scripts/05_run_visualization.py
scripts/06_train_xgboost.py
scripts/07_organize_experiment.py
scripts/08_train_cnn.py
tests/...
README.md
AGENTS.md
```

## Non-Goals

This change does not:

- support TUAB in the CNN pipeline;
- reproduce the thesis MFCC, HMM, or CNN-MLP systems;
- change the established 211- or 291-dimensional feature profiles;
- assign one permanent normal/abnormal label to a TUAB patient;
- aggregate TUAB probabilities at patient level;
- parse seizure annotations for TUAB;
- alter the established TUEP label encoding.
