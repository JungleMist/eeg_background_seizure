# Parallel Profile Feature Extraction Design

## Goal

Make the feature-profile path used by `scripts/06_train_xgboost.py` extract
features with `ProcessPoolExecutor`, matching the file-level parallelism of the
legacy `build_dataset()` path while preserving the current feature values,
sample metadata, cache layout, and deterministic ordering.

## Scope

- Parallelize `build_dataset_with_profile()` at one task per NPZ cache file.
- Add `--workers N` to script 06 and pass it to feature extraction.
- Support both `base211` and `base211_conn80` profiles.
- Keep condition training sequential and leave XGBoost/GridSearchCV parallelism
  unchanged.
- Do not change feature definitions, schema hashes, cache filenames, or model
  outputs.

## Design

Add a module-level, pickle-safe helper in `eeg_bg/features/extraction.py`. The
helper receives only serializable arguments, loads one NPZ file, filters it by
split, extracts all epochs through the selected profile, validates every output
dimension, and returns the file index plus its feature rows, labels, and subject
IDs. It retains the current behavior of returning no rows for an unreadable or
invalid cache file.

`build_dataset_with_profile()` will submit all files to a
`ProcessPoolExecutor(max_workers=max_workers)`, collect completed futures with a
progress bar, and merge successful results by original sorted file index. This
keeps output ordering deterministic even though workers finish out of order.
Worker exceptions that escape the helper will continue to propagate through
`future.result()` instead of being silently lost at the executor boundary.

Script 06 will accept a positive `--workers` integer. Omitting it passes `None`,
which lets `ProcessPoolExecutor` choose its normal CPU-count-based default. The
value flows through `main()`, `run_condition()`, `_load_or_extract_features()`,
and finally `build_dataset_with_profile()`. When an existing feature cache is
reused, no worker pool is created.

## Validation

- Unit-test both profiles with `max_workers=2`.
- Verify dimensions, labels, subject IDs, split filtering, and deterministic
  file/epoch ordering.
- Verify the CLI exposes `--workers` and rejects non-positive values.
- Run the focused feature extraction tests, the script CLI help check, and then
  the complete non-integration test suite if focused tests pass.

