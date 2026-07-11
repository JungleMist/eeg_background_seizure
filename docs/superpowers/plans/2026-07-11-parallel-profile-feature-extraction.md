# Parallel Profile Feature Extraction Implementation Plan

**Goal:** Use file-level `ProcessPoolExecutor` parallelism for the profile-based
feature extraction path used by script 06, with user-controllable worker count.

**Architecture:** A pickle-safe module-level worker extracts one NPZ file. The
parent process submits files concurrently and merges results by original file
index so concurrency does not make dataset ordering nondeterministic.

**Tech stack:** Python 3.11, `concurrent.futures`, NumPy, pytest, existing conda
environment `eeg_pipeline`.

## File map

| Action | File | Responsibility |
|---|---|---|
| Modify | `eeg_bg/features/extraction.py` | Profile worker and parallel orchestration |
| Modify | `scripts/06_train_xgboost.py` | `--workers` CLI and argument propagation |
| Modify | `tests/test_features/test_extraction.py` | Profile multiprocessing regression tests |
| Add or modify | `tests/` script-06 test module, if an existing suitable module is found | CLI validation test |
| Modify | `AGENTS.md` | Document the new script 06 option and actual parallel behavior |

## Task 1: Establish regression tests

- [ ] Import `build_dataset_with_profile` in the feature extraction tests.
- [ ] Create at least two NPZ files with multiple epochs and deliberately
      distinguishable subject IDs/labels.
- [ ] Add a `base211` test using `max_workers=2` that asserts `(n, 211)` output,
      aligned labels/IDs, split filtering, and sorted-file/within-file ordering.
- [ ] Add a `base211_conn80` smoke test using `max_workers=2` that asserts
      `(n, 291)` and finite values.
- [ ] Add or extend a script CLI test for `--workers 2` parsing and rejection of
      `--workers 0`/negative values.
- [ ] Run the new tests first and confirm they fail for the expected missing
      implementation or CLI behavior.

## Task 2: Implement a pickle-safe per-file profile worker

- [ ] Add a module-level `_extract_one_file_with_profile(args)` helper; do not
      use a nested function or lambda because macOS/Windows process spawning
      must pickle the callable.
- [ ] Pass the file index, path, cache array key, split, sampling parameters,
      profile name, and connectivity `nperseg` as plain serializable values.
- [ ] Resolve `PROFILES[profile_name]` inside the worker process.
- [ ] Load the NPZ, reject non-matching splits, apply the standard-19 fallback,
      extract all epochs, and validate `profile.dim` exactly as the serial code
      currently does.
- [ ] Return `(file_index, rows, labels, subject_ids)`.
- [ ] Preserve the current bad-file policy by returning an empty result for
      exceptions raised while loading or extracting that NPZ.

## Task 3: Parallelize `build_dataset_with_profile()`

- [ ] Build one argument tuple per sorted NPZ path.
- [ ] Create `ProcessPoolExecutor(max_workers=max_workers)` and submit one future
      per file.
- [ ] Consume completions through `as_completed()` so the progress bar advances
      as workers finish.
- [ ] Store each result under its file index rather than immediately appending.
- [ ] Merge results in ascending file-index order, preserving the old deterministic
      sample ordering and epoch order within each file.
- [ ] Retain the current empty-array shapes and dtypes for empty splits.
- [ ] Update the docstring to describe `max_workers=None` and file-level process
      parallelism.

## Task 4: Expose `--workers` in script 06

- [ ] Add a small positive-integer argparse validator, or equivalent validation,
      so zero and negative worker counts fail before extraction starts.
- [ ] Add `--workers N`, defaulting to `None`; explain that omission uses the
      executor default and that cached features bypass extraction.
- [ ] Thread `workers` through `main()`, `run_condition()`, and
      `_load_or_extract_features()` into `build_dataset_with_profile(max_workers=...)`.
- [ ] Ensure every train/val/test extraction and every selected condition receives
      the same worker limit.
- [ ] Leave cached-feature loading, schema hashes, condition sequencing, XGBoost
      `n_jobs`, and CUDA behavior untouched.

## Task 5: Documentation consistency

- [ ] Update the script usage/docstring with a `--workers` example.
- [ ] Update `AGENTS.md` so script 06 lists `[--workers N]` and accurately says
      profile feature extraction uses `ProcessPoolExecutor` while conditions run
      sequentially.
- [ ] Avoid unrelated README or configuration edits unless tests reveal a direct
      inconsistency.

## Task 6: Verification

- [ ] Run focused tests:

```bash
conda run -n eeg_pipeline python -m pytest tests/test_features/test_extraction.py -v
```

- [ ] Verify CLI help:

```bash
conda run -n eeg_pipeline python scripts/06_train_xgboost.py --help
```

- [ ] Verify invalid worker counts fail cleanly:

```bash
conda run -n eeg_pipeline python scripts/06_train_xgboost.py --workers 0
```

- [ ] Run all non-integration tests:

```bash
conda run -n eeg_pipeline python -m pytest tests/ -m "not integration"
```

- [ ] Inspect `git diff --check` and `git diff` to confirm the patch is surgical.

## Acceptance criteria

- Fresh profile extraction visibly runs multiple worker processes when more than
  one NPZ file is available and `--workers` is greater than one or omitted on a
  multi-core machine.
- `--workers 1` remains a supported low-resource execution mode.
- Both feature profiles produce the same values, dimensions, labels, IDs, and
  deterministic ordering as the prior serial implementation.
- Existing feature caches are loaded without spawning extraction workers.
- No changes occur to XGBoost GPU scheduling or cross-condition sequencing.
- Focused and non-integration tests pass.

