# Tabular baselines

Three arms — XGBoost for binary classification, DeepHit for time-to-event, and a most-recent-value
baseline for regression — over two feature sets selected at extraction time.

## The two feature sets

`ora.featurizers.CountFeaturizer` under default flags (`numeric_value_decile=False`) drops every event
carrying a numeric value from the feature matrix entirely — not only the value but the occurrence count
— because `add_preprocess_data` routes numeric events into a branch that is a no-op when the flag is
off, so the code never reaches `observed_codes` and no column is allocated. Under that feature set the
models never learn that a lab was even ordered.

`--value_stats yes` (the default) restores that information by reproducing the **MEDS-Tab**
(arXiv:2411.00200) aggregation set:

```
code/count, value/count, value/sum, value/sum_sqd, value/min, value/max
```

over nested look-back windows `[7d, full]` — 12 columns per numeric code, for each numeric code
occurring at least 10 times. No `last`, no recency, no derived mean/std, no per-code standardization:
the feature specification is the published one.

`--value_stats no` extracts code counts only, giving the count-only baseline. The two write to separate
roots (`meds_tab` and `meds_tab_count`) so both can exist side by side and both rows can be reported.

## Files

**`run_all.py` is the only entry point.** Everything else is either a library it imports or a
feature-extraction step that runs before it.

| File | Kind | Purpose |
|---|---|---|
| `run_all.py` | **entry point** | `--task {classification,tte,regression,all}` |
| `feature_extraction/meds_tab_featurizer.py` | library | `MedsTabNumericFeaturizer` |
| `feature_extraction/generate_features.py` | entry point | Feature extraction; `--value_stats {yes,no}` |
| `feature_extraction/extract.sh` | entry point | All nine cohorts (3 TTE + 3 phenotype-binary + 3 outcome) |
| `feature_extraction/validate_featurizer.py` | entry point | Brute-force correctness check — run this first |
| `common.py` | library | Seeding, metric IO, sparse/dense feature preparation |
| `classification_xgboost.py` | library | XGBoost arm: `split_binary`, `train_binary`, AUROC bootstraps |
| `tte_deephit.py` | library | DeepHit arm: `split_survival`, `train_survival`, IPCW C-index and IBS |
| `regression_most_recent_impute.py` | library + CLI | Most-recent-value baseline |
| `collect_results.py` | entry point | Merges per-task metric JSON into result tables |
| `collect_compute.py` | entry point | Merges per-task runtime / peak-RSS JSON into a compute table |

## Run order

Extraction needs `meds_reader` and `ora`. `run_all.py` additionally needs `xgboost` and
`scikit-survival`.

Everything runs with `-m` from the repository root. `BASE=ora.omop_meds_tutorial.evaluation.baseline`.

```bash
# 1. Validate the featurizer against a brute-force reference. Do not skip.
python -m $BASE.feature_extraction.validate_featurizer --num_subjects 200

# 2. Extract features for all nine cohorts
bash src/ora/omop_meds_tutorial/evaluation/baseline/feature_extraction/extract.sh
VALUE_STATS=no bash .../feature_extraction/extract.sh     # count-only feature set

# 3. Train
python -m $BASE.run_all --task all --n_jobs 24 --truncation 300
python -m $BASE.run_all --task tte --phenotypes ami       # one arm, one task
python -m $BASE.run_all --task regression \
    --cohort_dir ... --meds_path ... --split_path ...
python -m $BASE.run_all --task all --data_path <ROOT>/meds_tab_count   # count-only row

# 4. Tables
python -m $BASE.collect_results
python -m $BASE.collect_compute --csv compute.csv
```

`--truncation` must match the value the foundation-model probe is run with
(`classification_tte/run.sh --truncation`), or the two arms are scored at different horizons and their
C-indices are not comparable. Both default to 300 (`common.DEFAULT_TRUNCATION`). Dataset paths at the
top of `feature_extraction/extract.sh` and `run_all.py` are placeholders — point them at your MEDS
reader, cohort and split files first.

## Value-aware vs count-only

Cohorts, splits, metrics and the bootstrap protocol are identical between the two feature sets. The
differences are:

1. **Features.** `MedsTabNumericFeaturizer` is appended to the featurizer list. `CountFeaturizer` is
   untouched, so non-numeric codes are featurized exactly the same way in both.
2. **Output paths.** Separate feature and result roots, so both rows can be reported side by side.
   Train/tuning/hyper resampling uses the same seeds, so the two rows are directly comparable.
3. **Sparse loading.** `load_features_labels` aligns rows through the MultiIndex and row-slices the CSR
   matrix rather than densifying. The value-aware matrix is far wider and the dense path risks OOM.
   XGBoost consumes CSR directly; DeepHit still receives dense, scaled input with all-zero training
   columns removed.
4. **Hyperparameter search.** Both arms are widened to match or exceed the foundation-model probes they
   are compared against:

   | | FM probe | Count-only baseline | Here |
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
