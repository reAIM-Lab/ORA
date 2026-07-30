"""Binary-classification arm of the tabular baseline: XGBoost training and AUROC scoring."""
import json
import os
import pickle

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from meds_evaluation.evaluate import evaluate_bootstrapped_binary_classification
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from tqdm import tqdm

from ..common.survival_utils import generate_bootstrap_indices
from .common import N_REPORT, N_SAVE, set_seed, split_sparse


# =========================
# Binary pipeline
# =========================
def split_binary(data, labels):
    split = labels["split_name"].values
    y = labels["boolean_value"].values

    mask = {
        "train": split == "train",
        "val": split == "tuning",
        "hyper": split == "hyper",
        "test": split == "test",
    }

    # XGBoost consumes the CSR matrix directly -- no densify, no scaling.
    X = split_sparse(data, mask)

    return X["train"], y[mask["train"]], X["val"], y[mask["val"]], X["hyper"], y[mask["hyper"]], X["test"], y[mask["test"]], labels.index[mask["test"]]


def evaluate_binary_meds(pred_df: pd.DataFrame, bootstrapping: int = 100) -> dict:
    """Bootstrapped binary-classification metrics via the meds_evaluation package.

    This is the reported number: `mean_roc_auc_score` +- `std_roc_auc_score` over
    `bootstrapping` resamples. meds_evaluation resamples through its own `_resample`
    helper rather than a plain row bootstrap, so use it for every arm (foundation
    models and this baseline) or the error bars are not comparable.

    Note: meds_evaluation covers **binary classification only**. There is no
    survival/C-index entry point in the package, so time-to-event tasks keep the
    `concordance_index_ipcw` bootstrap in `evaluate_survival`.
    """
    df = pl.DataFrame(
        {
            "subject_id": pred_df["subject_id"].to_numpy().astype("int64"),
            "prediction_time": pd.to_datetime(pred_df["prediction_time"]).to_numpy(),
            "boolean_value": pred_df["boolean_value"].to_numpy().astype(bool),
            "predicted_boolean_value": pred_df["predicted_boolean_value"].to_numpy().astype(bool),
            "predicted_boolean_probability": pred_df["predicted_boolean_probability"]
            .to_numpy()
            .astype("float64"),
        }
    ).with_columns(pl.col("prediction_time").cast(pl.Datetime))

    res = evaluate_bootstrapped_binary_classification(df, bootstrapping=bootstrapping)
    print(f"MEDS AUROC: {res['mean_roc_auc_score']:.4f} +- {res['std_roc_auc_score']:.4f}"
          f"  (n={bootstrapping} bootstraps)")
    return res


def evaluate_binary(pred, y, n_report=N_REPORT, n_save=N_SAVE, test_order=None,
                    save_path=None, model_name="meds_tab_xgboost", best_params=None):
    """AUROC / AUPRC bootstraps, following linear_probe.logistic_regression exactly.

    * **reported** -- `n_report` (100) resamples with `np.random.default_rng(0)` and
      `rng.integers(0, n, size=n)`, skipping degenerate single-class draws, std with
      `ddof=1`. Byte-identical procedure to the FM arms.
    * **saved** -- `n_save` (10,000) resamples from the shared
      `generate_bootstrap_indices(n, 10000, seed=42)` helper, written to
      `binary_results_{model_name}.pkl` alongside `test_order`, so binary tasks can be
      paired-tested the same way `common.survival_utils.paired_bootstrap_test` handles TTE.

    NOTE: `linear_probe.logistic_regression` currently computes only the 100 reported
    resamples and discards them. To paired-test binary tasks against the FM arms, the
    same `generate_bootstrap_indices` save needs to be added there too -- otherwise the
    FM side has no per-resample AUROC to pair against.
    """
    y_true = np.asarray(y, dtype=np.int64)
    y_pred = np.asarray(pred, dtype=float)
    n = len(y_true)

    def _auroc_auprc(idx):
        yb, pb = y_true[idx], y_pred[idx]
        if np.unique(yb).size < 2:
            return None
        precision, recall, _ = precision_recall_curve(yb, pb)
        return roc_auc_score(yb, pb), auc(recall, precision)

    # ---- reported: 100 resamples, matching linear_probe ----
    results = {"AUROC": [], "AUPRC": []}
    rng = np.random.default_rng(0)
    for _ in range(n_report):
        got = _auroc_auprc(rng.integers(0, n, size=n))
        if got is None:
            continue
        results["AUROC"].append(got[0])
        results["AUPRC"].append(got[1])

    print(f"AUROC: {np.mean(results['AUROC']):.4f} "
          f"({np.std(results['AUROC'], ddof=1):.4f})  [n={len(results['AUROC'])} of {n_report}]")
    print(f"AUPRC: {np.mean(results['AUPRC']):.4f} "
          f"({np.std(results['AUPRC'], ddof=1):.4f})")

    # ---- saved: 10,000 shared-index resamples for later paired testing ----
    saved_auroc, saved_auprc = [], []
    for idx in tqdm(generate_bootstrap_indices(n, n_save, seed=42), desc="bootstrap (saved)"):
        got = _auroc_auprc(idx)
        if got is None:
            continue
        saved_auroc.append(got[0])
        saved_auprc.append(got[1])

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, f"binary_results_{model_name}.pkl")
        with open(out, "wb") as f:
            pickle.dump(
                {
                    "AUROC": saved_auroc,
                    "AUPRC": saved_auprc,
                    "test_order": test_order,
                    "model_name": model_name,
                    "best_params": best_params,
                },
                f,
            )
        with open(os.path.join(save_path, f"binary_metrics_{model_name}.json"), "w") as f:
            json.dump(
                {
                    "mean_auroc": float(np.mean(saved_auroc)),
                    "std_auroc": float(np.std(saved_auroc)),
                    "n_bootstrap": len(saved_auroc),
                    "model_name": model_name,
                    "best_params": best_params,
                },
                f, indent=2, default=str,
            )
        print(f"saved {len(saved_auroc)} AUROC resamples to {out}")

    results["_saved_bootstrap"] = {
        "n_save": len(saved_auroc),
        "mean_auroc": float(np.mean(saved_auroc)),
        "std_auroc": float(np.std(saved_auroc)),
        "path": str(save_path) if save_path else None,
    }
    return results


def train_binary(x_train, y_train, x_val, y_val, x_hyper, y_hyper, x_test, y_test, test_index,
                 n_jobs=-1, save_path=None, model_name="meds_tab_xgboost", device="cpu"):
    set_seed()

    # Widened from the submitted baseline (2 configurations) toward the search
    # budget the MEDS-Tab recipe uses (100 Optuna trials). 24 configurations with
    # early stopping on the held-out `hyper` split.
    grid = ParameterGrid({
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 6, 10],
        "n_estimators": [200, 1000],
        "subsample": [0.8],
        "colsample_bytree": [0.5, 1.0],
    })

    best_model, best_auc, best_params = None, -np.inf, None

    for p in grid:
        print(f"p is {p}")
        model = xgb.XGBClassifier(
            **p,
            tree_method="hist",
            # device="cuda" is ~9x faster than 32 CPU threads on this matrix
            # (13.9s vs 126.1s per config at 124,734 x 80,707, 217M nnz) with
            # equivalent AUROC (0.8392 vs 0.8368; they differ only in where early
            # stopping fired). Same algorithm -- `hist` either way.
            device=device,
            n_jobs=n_jobs,
            eval_metric="auc",
            # Constructor arg requires xgboost >= 1.6; passing it to .fit() was
            # removed in xgboost 2.0.
            early_stopping_rounds=20,
        )
        model.fit(x_train, y_train, eval_set=[(x_hyper, y_hyper)], verbose=False)

        val_pred = model.predict_proba(x_val)[:, 1]
        auc = roc_auc_score(y_val, val_pred)

        if auc > best_auc:
            best_auc, best_model, best_params = auc, model, p

    print(f"best XGBoost config: {best_params} (val AUROC {best_auc:.4f})")
    test_pred = best_model.predict_proba(x_test)[:, 1]
    # test_order must match the FM arms for paired testing (see evaluate_binary).
    test_order = [
        (int(s_), pd.Timestamp(t_).isoformat())
        for s_, t_ in zip(
            test_index.get_level_values("subject_id"),
            test_index.get_level_values("prediction_time"),
        )
    ]
    metrics = evaluate_binary(
        test_pred, y_test, test_order=test_order, save_path=save_path,
        model_name=model_name, best_params=best_params,
    )
    metrics["_best_params"] = [{**best_params, "val_auroc": float(best_auc)}]

    pred_df = pd.DataFrame({
        "subject_id": test_index.get_level_values("subject_id"),
        "prediction_time": test_index.get_level_values("prediction_time"),
        "predicted_boolean_probability": test_pred,
        # meds_evaluation requires a thresholded prediction alongside the probability.
        "predicted_boolean_value": test_pred >= 0.5,
        "boolean_value": y_test,
    })

    # The reported metric: meds_evaluation bootstrapped mean +- std.
    metrics["_meds_evaluation"] = evaluate_binary_meds(pred_df)

    return metrics, pred_df