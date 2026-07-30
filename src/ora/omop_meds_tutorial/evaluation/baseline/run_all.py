"""The single entry point for the tabular baselines.

One command for all three task families; `--task` selects which arms run.

    RUN=ora.omop_meds_tutorial.evaluation.baseline.run_all

    python -m $RUN --task all                        # classification + tte
    python -m $RUN --task classification --n_jobs 24 # XGBoost, CPU-bound
    python -m $RUN --task tte --phenotypes ami       # DeepHit, GPU-bound
    python -m $RUN --task regression \
        --cohort_dir ... --meds_path ... --split_path ...

| task           | model      | arm                                    |
|----------------|------------|----------------------------------------|
| classification | XGBoost    | classification.py                      |
| tte            | DeepHit    | tte.py                                 |
| regression     | last value | regression.py                          |

Features come from `feature_extraction/`. Run that first. Whether they carry the
MEDS-Tab numeric aggregations is decided there by `--value_stats`, not here, so
point `--data_path` at the matching feature root (`meds_tab` or `meds_tab_count`).

`--truncation` must match the horizon the foundation-model probe is run with
(`run.sh --truncation`), or the C-indices are not comparable.

The DeepHit arm is GPU-bound, so the three TTE tasks can run concurrently on
separate idle GPUs -- a free 3x on wall-clock -- while XGBoost runs on CPU:

    CUDA_VISIBLE_DEVICES=2 python -m $RUN --task tte --phenotypes ami    --n_jobs 8 &
    CUDA_VISIBLE_DEVICES=3 python -m $RUN --task tte --phenotypes stroke --n_jobs 8 &
    CUDA_VISIBLE_DEVICES=4 python -m $RUN --task tte --phenotypes masld  --n_jobs 8 &
    python -m $RUN --task classification --n_jobs 24
"""

import argparse
import json
import os
import pickle
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .classification import split_binary, train_binary
from .common import DEFAULT_TRUNCATION, load_metrics, save_metrics, summarize
from .regression import run_baseline as run_regression_baseline
from .tte import split_survival, train_survival

ROOT = "/path/to/workspace/ora"
DATA_PATH = os.path.join(ROOT, "meds_tab")
RESULTS_PATH = os.path.join(ROOT, "result", "meds_tab")
SPLIT_PATH = os.path.join(ROOT, "metadata", "main_split.csv")
TTE_COL = "time_to_event_days"

PHENOTYPES = ["ami", "stroke", "masld"]
BINARY_ONLY = ["in_hospital_mortality", "readmission", "long_los"]

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="all",
                    choices=["classification", "tte", "regression", "all"],
                    help="Which arm(s) to run. 'all' runs classification + tte; regression "
                         "needs --cohort_dir/--meds_path/--split_path and is opt-in.")
parser.add_argument("--data_path", default=None,
                    help="Feature root. Defaults to <ROOT>/meds_tab (value-aware). Point at "
                         "<ROOT>/meds_tab_count for features extracted with --value_stats no.")
parser.add_argument("--phenotypes", nargs="*", default=None,
                    help=f"Subset of {PHENOTYPES}")
parser.add_argument("--binary_only", nargs="*", default=None,
                    help=f"Subset of {BINARY_ONLY}")
parser.add_argument("--n_jobs", type=int, default=16,
                    help="CPU cores for XGBoost (-1 = all). Report this alongside timings.")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--results_dir", default=None,
                    help="Override the results directory (use a fresh one to avoid "
                         "overwriting a previous sweep).")
parser.add_argument("--wd", default=None,
                    help="Comma-separated weight-decay values, overriding the default "
                         "[0.0, 1e-2]. e.g. --wd 1e-4 gives a single value (24 configs).")
parser.add_argument("--deephit_grid", default="fast", choices=["fast", "full"],
                    help="'fast' = 48 configs (drops the [50,50] architecture); 'full' = 72.")
parser.add_argument("--device", default="cpu",
                    help="XGBoost device: 'cpu' or 'cuda'. cuda is ~9x faster here.")
parser.add_argument("--truncation", type=float, default=DEFAULT_TRUNCATION,
                    help="Evaluation horizon in days for the TTE arm. Must match the value the "
                         "foundation-model probe is run with (run.sh --truncation), or the two "
                         "arms are not comparable.")
parser.add_argument("--patience", type=int, default=5,
                    help="EarlyStopping patience for DeepHit (torchtuples default is 10).")
# --- regression arm only ---
parser.add_argument("--cohort_dir", default=None, help="[--task regression] cohort directory")
parser.add_argument("--meds_path", default=None, help="[--task regression] MEDS reader")
parser.add_argument("--split_path", default=None, help="[--task regression] main_split.csv")
parser.add_argument("--dataset", default="mimic", choices=["mimic", "institutional"],
                    help="[--task regression] which target-code map to use")

# Populated by main(). Importing this module must not parse argv or run a sweep.
args = None
WD_VALUES = None


def peak_rss_gb() -> float:
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(self_kb, child_kb) / 1024**2


def load_features_labels(feature_path, label_path, split_path):
    """Align the sparse feature matrix to the label table without densifying.

    The original implementation built a dense DataFrame via `.todense()` and used
    `.loc`. With the value-aware feature set that matrix is far larger, so rows
    are aligned through the MultiIndex and the CSR matrix is row-sliced instead.

    Duplicate (subject_id, prediction_time) keys are resolved keep="last" on both
    sides, matching the original `remove_duplicates` behaviour. The train/tuning/
    hyper resampling is byte-identical to the submitted baseline (same seeds), so
    the two rows are directly comparable.
    """
    data = pickle.load(open(feature_path, "rb"))
    features = data["features"].tocsr()

    idx = pd.MultiIndex.from_arrays(
        [data["subject_ids"], data["feature_times"]],
        names=["subject_id", "prediction_time"],
    )
    keep = ~idx.duplicated(keep="last")
    features = features[np.flatnonzero(keep)]
    idx = idx[keep]

    labels = pd.read_parquet(label_path).set_index(["subject_id", "prediction_time"])
    labels = labels[~labels.index.duplicated(keep="last")]

    pos = idx.get_indexer(labels.index)
    missing = int((pos < 0).sum())
    if missing:
        print(f"  WARNING: {missing}/{len(labels)} labels have no feature row; dropping them")
        labels = labels[pos >= 0]
        pos = pos[pos >= 0]
    X = features[pos]

    split = pd.read_csv(split_path, index_col="subject_id")
    labels = labels.join(split)

    val = labels[labels.split_name == "train"].index.to_series().sample(
        frac=0.2, replace=False, random_state=42
    )
    hyper = val.sample(frac=0.5, replace=False, random_state=42)
    labels.loc[val, "split_name"] = "tuning"
    labels.loc[hyper, "split_name"] = "hyper"

    print(f"  features {X.shape}, nnz={X.nnz}, labels {len(labels)}")
    return X, labels


def run_one(name: str, suffix: str, model: str, outcome: str) -> None:
    out_file = os.path.join(RESULTS_PATH, f"{name}_{model}.pickle")
    print("-" * 42)
    print(f"mimic {name} [{outcome}]")

    if os.path.exists(out_file) and not args.overwrite:
        summarize(load_metrics(out_file))
        return

    try:
        base_dir = os.path.join(DATA_PATH, f"{name}{suffix}")
        t0 = time.perf_counter()
        X, y = load_features_labels(
            os.path.join(base_dir, "features", f"{name}.pkl"),
            os.path.join(base_dir, "labels", f"{name}.parquet"),
            SPLIT_PATH,
        )
        t_load = time.perf_counter() - t0

        # Per-task directory for the 10,000-resample artifacts used by
        # common.survival_utils.paired_bootstrap_test, mirroring the FM arms' layout.
        boot_dir = os.path.join(RESULTS_PATH, f"{name}_{model}_bootstrap")

        t1 = time.perf_counter()
        if outcome == "survival":
            # Collects the column subset + scaler needed to re-apply the saved net.
            artifacts = {"feature_path": os.path.join(base_dir, "features", f"{name}.pkl"),
                         "label_path": os.path.join(base_dir, "labels", f"{name}.parquet"),
                         "split_path": SPLIT_PATH, "tte_col": TTE_COL}
            splits = split_survival(X, y, TTE_COL, artifacts=artifacts)
            # test_order must match the FM arms exactly: (subject_id, prediction_time)
            # in test-row order. paired_bootstrap_test asserts equality of this list.
            test_idx = y.index[y["split_name"].values == "test"]
            test_order = [
                (int(s_), pd.Timestamp(t_).isoformat())
                for s_, t_ in zip(
                    test_idx.get_level_values("subject_id"),
                    test_idx.get_level_values("prediction_time"),
                )
            ]
            metrics = train_survival(
                *splits, model_type="deephit",
                test_order=test_order, save_path=boot_dir,
                model_name=f"meds_tab_deephit_{name}",
                grid_mode=args.deephit_grid, patience=args.patience,
                n_jobs=args.n_jobs, artifacts=artifacts, wd_values=WD_VALUES,
                truncation=args.truncation,
            )
        else:
            splits = split_binary(X, y)
            metrics, pred_df = train_binary(
                *splits, n_jobs=args.n_jobs, save_path=boot_dir,
                model_name=f"meds_tab_xgboost_{name}", device=args.device,
            )
            pred_df.to_parquet(out_file.replace(".pickle", "_predictions.parquet"), index=False)
        t_train = time.perf_counter() - t1

        save_metrics(metrics, out_file)
        with open(out_file.replace(".pickle", "_compute.json"), "w") as f:
            json.dump(
                {
                    "task": name,
                    "outcome": outcome,
                    "model": model,
                    "n_rows": int(X.shape[0]),
                    "n_columns": int(X.shape[1]),
                    "nnz": int(X.nnz),
                    "load_seconds": round(t_load, 1),
                    "train_and_tune_seconds": round(t_train, 1),
                    "peak_rss_gb": round(peak_rss_gb(), 2),
                    "n_jobs": args.n_jobs,
                    "device": args.device if outcome == "binary" else "cuda",
                },
                f,
                indent=2,
            )
        print(f"  load {t_load:.1f}s, train+tune {t_train:.1f}s, peak RSS {peak_rss_gb():.1f} GB")

    except Exception as e:  # noqa: BLE001 -- keep the sweep going
        print(f"[ERROR] mimic | {name} | {outcome}: {type(e).__name__}: {e}")


def main():
    global args, WD_VALUES, DATA_PATH, RESULTS_PATH
    args = parser.parse_args()
    if args.data_path:
        DATA_PATH = args.data_path
    if args.results_dir:
        RESULTS_PATH = args.results_dir
    WD_VALUES = [float(x) for x in args.wd.split(",")] if args.wd else None

    if args.task == "regression":
        missing = [f for f in ("cohort_dir", "meds_path", "split_path") if not getattr(args, f)]
        if missing:
            parser.error("--task regression requires " + ", ".join("--" + m for m in missing))
        run_regression_baseline(
            cohort_dir=Path(args.cohort_dir),
            meds_path=args.meds_path,
            split_path=Path(args.split_path),
            output_dir=Path(RESULTS_PATH),
            dataset=args.dataset,
        )
        return

    os.makedirs(RESULTS_PATH, exist_ok=True)
    phenotypes = args.phenotypes if args.phenotypes is not None else PHENOTYPES
    binary_only = args.binary_only if args.binary_only is not None else BINARY_ONLY
    run_tte = args.task in ("tte", "all")
    run_cls = args.task in ("classification", "all")

    # Phenotypes carry both a survival and a binary label.
    for name in phenotypes:
        if run_tte:
            run_one(name, "", "deephit", "survival")
        if run_cls:
            run_one(name, "_binary", "xgboost", "binary")

    # Patient outcomes are binary only.
    if run_cls:
        for name in binary_only:
            run_one(name, "_binary", "xgboost", "binary")


if __name__ == "__main__":
    main()
