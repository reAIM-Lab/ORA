# Value-aware tabular baseline (MEDS-Tab aggregation set)

The tabular baseline in `../baseline/` builds features with `ora.featurizers.CountFeaturizer` under
default flags (`numeric_value_decile=False`). Under those defaults every event carrying a numeric value
is dropped from the feature matrix entirely — not only the value but the occurrence count — because
`add_preprocess_data` routes numeric events into a branch that is a no-op when the flag is off, so the
code never reaches `observed_codes` and no column is allocated. The tree and DeepHit baselines therefore
never learn that a lab was even ordered.

This package restores that information by reproducing the **MEDS-Tab** (arXiv:2411.00200) aggregation
set:

```
code/count, value/count, value/sum, value/sum_sqd, value/min, value/max
```

over nested look-back windows `[7d, full]` — 12 columns per numeric code, for each numeric code
occurring at least 10 times. No `last`, no recency, no derived mean/std, no per-code standardization:
the feature specification is the published one.

## Files

| File | Purpose |
|---|---|
| `meds_tab_featurizer.py` | `MedsTabNumericFeaturizer` — the featurizer |
| `generate_tabular_features.py` | Feature-extraction entry point |
| `extract.sh` | Extraction over all nine cohorts (3 TTE + 3 phenotype-binary + 3 outcome) |
| `validate_featurizer.py` | Brute-force correctness check — run this first |
| `train_utils.py` | `../baseline/train_utils.py` plus widened grids and sparse handling |
| `run_all.py` | Driver: sparse for XGBoost, dense + scaled for DeepHit |
| `collect_results.py` | Merges per-task metric JSON into result tables |
| `collect_compute.py` | Merges per-task runtime / peak-RSS JSON into a compute table |
| `rescore_tte.py` | Re-scores saved TTE predictions |
| `export_best_300.py`, `launch.sh` | Convenience wrappers |

## Run order

Extraction needs `meds_reader` and `ora`. `run_all.py` additionally needs `xgboost` and
`scikit-survival`.

```bash
cd src/ora/omop_meds_tutorial/evaluation/baseline_value

# 1. Validate the featurizer against a brute-force reference. Do not skip.
python -m ora.omop_meds_tutorial.evaluation.baseline_value.validate_featurizer --num_subjects 200

# 2. Extract features for all nine cohorts
bash extract.sh

# 3. Train: DeepHit for phenotype TTE, XGBoost for all binary tasks
python run_all.py --n_jobs 24

# 4. Compute table
python collect_compute.py --csv compute.csv
```

`run_all.py` uses `from train_utils import ...`, so run it as a script from inside this directory, not
with `-m`. Dataset paths at the top of `extract.sh` and `run_all.py` are placeholders — point them at
your MEDS reader, cohort and split files before running.

## What differs from `../baseline/`

Cohorts, splits, metrics and the bootstrap protocol are unchanged. The differences are:

1. **Features.** `MedsTabNumericFeaturizer` is added to the featurizer list. `CountFeaturizer` is
   untouched, so non-numeric codes are featurized exactly as before.
2. **Output paths.** Written to a separate results directory so both baseline rows can be reported
   side by side. Train/tuning/hyper resampling uses the same seeds, so the two rows are comparable.
3. **Sparse loading.** `load_features_labels` aligns rows through the MultiIndex and row-slices the CSR
   matrix rather than densifying. The value-aware matrix is far wider and the dense path risks OOM.
   XGBoost consumes CSR directly; DeepHit still receives dense, scaled input with all-zero training
   columns removed.
4. **Hyperparameter search.** Both arms are widened to match or exceed the foundation-model probes they
   are compared against:

   | | FM probe | `../baseline/` | Here |
   |---|---|---|---|
   | DeepHit architecture | `[]` (linear) | `[]`, `[50]`, `[50,50]` | `[]`, `[50]`, `[50,50]` |
   | DeepHit learning rate | {1e-3, 1e-4} | {1e-4} | {1e-3, 1e-4} |
   | DeepHit batch size | {1024, 2048} | {1024} | {1024, 2048} |
   | DeepHit epochs | 1000 | 100 | 1000 |
   | Binary probe | `LogisticRegressionCV` (10 C values, 5-fold) | XGBoost, 2 configs | XGBoost, 24 configs + early stopping |

   The FM probes are deliberately linear — that is what linear probing means.
5. **Compute accounting.** Extraction writes `<task>_extraction_stats.json` (wall-clock, peak RSS,
   column budget, density); training writes `<task>_<model>_compute.json` (load time, train+tune time,
   peak RSS, `n_jobs`).
6. **Fixes carried over.** The `continue` that skipped the binary arm of the phenotype loop is removed,
   so every phenotype yields both a TTE and a binary result. Cohort loading prefers
   `train/tuning/held_out.parquet` over a recursive glob, because the TTE cohort directories also
   contain a whole-cohort parquet that is the union of the three splits, which a plain glob would read
   twice.

## Scope

Following the MEDS-Tab specification, this featurizer aggregates numeric measurements only;
categorical result codes remain excluded (`string_value_combination=False`). The high-volume
text-valued codes in MIMIC-IV are numbers stored as strings (blood pressure as `"100/48"`, weight,
BMI), which one-hot encoding would expand without bound. A minority of text codes are genuinely
categorical and clinically meaningful — heart rhythm, ectopy type, pain level — and remain excluded.
