"""Shared plumbing for the tabular baselines: seeding, metric IO and feature preparation.

Task-specific training lives in `classification.py` (XGBoost) and `tte.py` (DeepHit);
`regression.py` is the most-recent-value baseline and does not use this module.
"""
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Report on 100 resamples (paper tables); persist 10,000 for later paired tests.
N_REPORT = 100
N_SAVE = 10000

# Every arm is evaluated at this horizon, in days. The foundation-model probe is run
# with the same value (`run.sh --truncation`), so the two arms stay comparable.
DEFAULT_TRUNCATION = 300


# ========================= 
# Reproducibility
# =========================
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# IO helpers
# =========================
def save_metrics(metrics, path):
    with open(path, "wb") as f:
        pickle.dump(metrics, f)


def load_metrics(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def summarize(metrics: dict):
    for k, v in metrics.items():
        if k.startswith("_"):
            # bookkeeping (e.g. the selected hyperparameters), not a bootstrap metric
            print(f"{k}: {v}")
            continue
        v = np.asarray(v)
        print(f"{k}: {v.mean():.3f} ({v.std():.3f})")


# =========================
# Preprocessing
# =========================
def remove_duplicates(data: pd.DataFrame, labels: pd.DataFrame):
    return data[~data.index.duplicated(keep="last")], labels[~labels.index.duplicated(keep="last")]


def scale_split(covariates, split_mask):
    scaler = StandardScaler().fit(covariates[split_mask["train"]])
    return {
        k: scaler.transform(covariates[v])
        for k, v in split_mask.items()
    }


def split_sparse(covariates, split_mask):
    """Row-slice a sparse matrix per split, without scaling.

    Gradient-boosted trees are invariant to monotone per-feature transforms, so
    the StandardScaler in `scale_split` is unnecessary for XGBoost -- and it
    cannot run on a sparse matrix with `with_mean=True` anyway. Keeping the
    matrix sparse is what makes the wider value-aware feature set tractable.
    """
    return {k: covariates[np.flatnonzero(v)] for k, v in split_mask.items()}


def densify_for_mlp(covariates, split_mask, artifacts=None):
    """Densify + scale for DeepHit, dropping columns that are constant in train.

    DeepHit is a torchtuples MLP and needs dense float32 input. All-zero (or
    constant) training columns carry no information and only inflate the dense
    matrix, so they are dropped first. Prints the final shape: if this is still
    too wide for memory, raise --min_support at extraction time rather than
    dropping a window.
    """
    train_rows = covariates[np.flatnonzero(split_mask["train"])]
    nnz_per_col = np.asarray((train_rows != 0).sum(axis=0)).ravel()
    keep = np.flatnonzero(nnz_per_col > 0)
    print(f"densify_for_mlp: keeping {len(keep)}/{covariates.shape[1]} columns")

    # float32 end to end. sklearn preserves the input dtype, and torch consumes float32
    # anyway, so going through float64 only doubled both the memory traffic and the peak
    # RSS (106 GB -> ~55 GB on the AMI TTE cohort) for no numerical benefit.
    scaler = StandardScaler()
    scaler.fit(np.asarray(train_rows[:, keep].todense(), dtype=np.float32))

    if artifacts is not None:
        # Everything needed to turn a raw feature matrix back into model input.
        # Without these the saved net is unusable: the column subset and the
        # standardisation are both fitted on this task's training split.
        artifacts["keep_columns"] = keep
        artifacts["n_columns_in"] = int(covariates.shape[1])
        artifacts["scaler_mean"] = scaler.mean_
        artifacts["scaler_scale"] = scaler.scale_

    return {
        k: scaler.transform(
            np.asarray(covariates[np.flatnonzero(v)][:, keep].todense(), dtype=np.float32)
        ).astype("float32")
        for k, v in split_mask.items()
    }


