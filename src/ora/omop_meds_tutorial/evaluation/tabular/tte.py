"""Time-to-event arm of the tabular baseline: DeepHit training and IPCW C-index scoring."""
import json
import os
import pickle

import numpy as np
import torchtuples as tt
from pycox.models import DeepHitSingle, CoxPH
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.model_selection import ParameterGrid
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from tqdm import tqdm

# The same helpers the foundation-model probe uses, so bootstrap indices are
# byte-identical across arms and `paired_bootstrap_test` applies directly.
from ..common.survival_utils import (
    bootstrap_cindex_parallel,
    generate_bootstrap_indices,
    make_structured_array,
    save_survival_results,
)
from .common import DEFAULT_TRUNCATION, N_REPORT, N_SAVE, densify_for_mlp, set_seed, summarize


# =========================
# Survival pipeline
# =========================
def split_survival(data, labels, tte_col, artifacts=None):
    split = labels["split_name"].values
    tte = labels[tte_col].values
    event = labels["tte_label"].values.astype(bool)

    mask = {
        "train": split == "train",
        "val": split == "tuning",
        "hyper": split == "hyper",
        "test": split == "test",
    }

    # DeepHit needs dense, scaled input; see densify_for_mlp.
    X = densify_for_mlp(data, mask, artifacts=artifacts)

    return (
        X["train"], (tte[mask["train"]], event[mask["train"]]),
        X["val"],   (tte[mask["val"]],   event[mask["val"]]),
        X["hyper"], (tte[mask["hyper"]], event[mask["hyper"]]),
        X["test"],  (tte[mask["test"]],  event[mask["test"]]),
    )


def evaluate_survival(surv_df, y_train, y_test, truncation=DEFAULT_TRUNCATION, n_report=N_REPORT,
                      n_save=N_SAVE, test_order=None, save_path=None,
                      model_name="meds_tab_deephit", best_params=None, model=None,
                      n_jobs=16):
    """C-index / integrated Brier score, following linear_probe.py exactly.

    Two bootstraps, matching the foundation-model arms:

    * **reported** -- `n_report` (100) resamples drawn with `np.random.default_rng(42)`,
      the same generator and call pattern as `linear_probe.tte_evaluation`'s `ibs_rng`.
      These give the mean (std) in the paper tables.
    * **saved** -- `n_save` (10,000) resamples from `generate_bootstrap_indices(n, 10000,
      seed=42)`, the shared helper. Because that helper is deterministic in `n_test`, the
      indices are identical across every model evaluated on the same test set, which is
      what makes `common.survival_utils.paired_bootstrap_test` valid against the FM results.

    `paired_bootstrap_test` asserts equality of `test_order`, so `test_order` must be the
    list of (subject_id, prediction_time) pairs in test-row order.
    """
    t_train, e_train = y_train
    t_test, e_test = y_test

    et_train = make_structured_array(t_train, e_train)
    et_test = make_structured_array(t_test, e_test)

    times = surv_df.index.to_numpy()

    # ---- saved: 10,000 shared-index resamples for later paired testing ----
    bootstrap_indices = generate_bootstrap_indices(len(et_test), n_save, seed=42)
    c_indices_saved = bootstrap_cindex_parallel(
        surv_df, et_train, et_test, truncation, bootstrap_indices, n_jobs
    )

    if save_path is not None:
        saved = {
            "C-Index": c_indices_saved,
            "test_order": test_order,
            "model_name": model_name,
            "best_params": best_params,
        }
        if model is not None:
            # Same filenames and payload as the FM arms (tte_results_/tte_model_/tte_metrics_).
            save_survival_results(saved, model, save_path, truncation)
        else:
            # save_survival_results calls model.save_net(); without a model, write the
            # results pickle and metrics json itself, keeping the identical layout.
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, f"tte_results_{truncation}.pkl"), "wb") as f:
                pickle.dump(saved, f)
            with open(os.path.join(save_path, f"tte_metrics_{truncation}.json"), "w") as f:
                json.dump(
                    {
                        "mean": float(np.mean(c_indices_saved)),
                        "std": float(np.std(c_indices_saved)),
                        "truncation": float(truncation),
                        "n_bootstrap": len(c_indices_saved),
                        "model_name": model_name,
                        "best_params": best_params,
                    },
                    f, indent=2, default=str,
                )

    # ---- reported: 100 resamples, identical procedure to linear_probe ----
    tau_idx = np.argmin(np.abs(times - truncation))
    eval_mask = (times >= et_test["t"].min()) & (times <= et_test["t"].max()) & (times <= truncation)
    times_eval = times[eval_mask]

    results = {"C-Index": [], "Int-Brier": []}
    rng = np.random.default_rng(42)
    for _ in tqdm(range(n_report), desc="bootstrap (reported)"):
        sample = rng.choice(len(et_test), len(et_test))
        risk = 1.0 - surv_df.iloc[tau_idx, sample].values
        results["C-Index"].append(
            concordance_index_ipcw(et_train, et_test[sample], risk, tau=truncation)[0]
        )
        results["Int-Brier"].append(
            integrated_brier_score(
                et_train, et_test[sample], surv_df.iloc[eval_mask, sample].T.values, times_eval
            )
        )

    summarize(results)
    results["_saved_bootstrap"] = {
        "n_save": len(c_indices_saved),
        "mean": float(np.mean(c_indices_saved)),
        "std": float(np.std(c_indices_saved)),
        "path": str(save_path) if save_path else None,
    }
    print(f"saved {len(c_indices_saved)} C-index resamples for paired testing "
          f"(mean {np.mean(c_indices_saved):.4f})")
    return results


def train_survival(x_train, y_train, x_val, y_val, x_hyper, y_hyper, x_test, y_test, model_type,
                   test_order=None, save_path=None, model_name="meds_tab_deephit",
                   grid_mode="fast", patience=5, n_jobs=16, artifacts=None,
                   wd_values=None, truncation=DEFAULT_TRUNCATION):
    set_seed()

    in_features = x_train.shape[1]
    out_features = 1
    labtrans = None

    if model_type == "deephit":
        labtrans = LabTransDiscreteTime(100)
        y_train_norm = labtrans.fit_transform(*y_train)
        y_val = labtrans.transform(*y_val)
        y_hyper = labtrans.transform(*y_hyper)
        out_features = labtrans.out_features
    else:
        y_train_norm = y_train

    # Tuning budget is a strict SUPERSET of the DeepHit head used for the
    # foundation-model probes (tte/grid_search.py:79-97),
    # so the tabular baseline cannot be called under-tuned relative to them:
    #
    #   FM probe:  dropout {0, 0.1} x lr {1e-3, 1e-4} x batch {1024, 2048}, 8 configs,
    #              1000 epochs, architecture hard-coded to []  (linear -- it is a probe)
    #   here:      the same three axes, plus [50] and [50,50] as options, 24 configs,
    #              the same 1000-epoch budget.
    #
    # The submitted baseline used a single lr (1e-4), a single batch (1024) and only
    # 100 epochs -- 10x less training than the probes it is compared against. That was
    # the real handicap; the architecture grid already included non-linear options.
    #
    # `wd` (decoupled weight decay, via tt.optim.AdamW) is the one axis the FM probe
    # does not have, and it is added because this arm needs it: a linear DeepHit over
    # ~80k standardised features with 100 duration bins is ~8M parameters against ~95k
    # training rows, and without an L2 penalty the validation loss rises from the first
    # epoch (train 4.04 -> 2.45 while val 6.76 -> 8.03 on a 49k-feature subset).
    # The FM probe sees 768 features (~77k parameters) and does not have this problem.
    # Regularising a *baseline* can only strengthen it, so this cannot be called unfair.
    # Architecture options are bottleneck *widths*, not depths. At ~74k inputs and 100
    # duration bins the first Linear is >99% of the parameters, so depth is nearly free
    # and nearly inert ([50] -> [50,50] adds 2,650 of 3.7M parameters) while width moves
    # capacity by millions:  [64] 4.8M  <  [] 7.4M  <  [128] 9.5M.
    # `[]` (no hidden layer, no nonlinearity) is the FM probe's architecture and must
    # stay: it is what makes this grid a superset of the probe's search space.
    #   fast -> 32 configs, full -> 48 configs.
    layer_options = [[], [50]] if grid_mode == "fast" else [[], [64], [128]]
    grid = ParameterGrid({
        "lr": [1e-3, 1e-4],
        "batch": [1024, 2048],
        "do": [0, 0.1],
        "layers": layer_options,
        # Two values, not three: this is where the 72 -> 48 config cut was taken,
        # because weight decay is the only axis the FM probe does not also search.
        # Keeping lr/batch/dropout at full resolution preserves the parity argument.
        # 1e-4 was dropped rather than 1e-2: at 4.8-9.5M parameters it is mild enough
        # to be near-indistinguishable from 0.0, so it would have spent half the grid
        # measuring the same regime twice.
        "wd": wd_values if wd_values is not None else [0.0, 1e-2],
    })
    print(f"DeepHit grid: {len(grid)} configs (mode={grid_mode})")
    # Matches `epochs = 1000` in the FM probe. EarlyStopping still cuts most
    # configurations short; torchtuples places the model on GPU automatically when one
    # is available, which is what makes this budget affordable at ~80k input features.
    epochs = 1000

    best_model, best_loss, best_params = None, np.inf, None

    for p in grid:
        net = tt.practical.MLPVanilla(
            in_features, p["layers"], out_features, batch_norm=True, dropout=p['do']
        )

        # AdamW with decoupled weight decay; wd=0.0 reproduces plain Adam.
        optimizer = tt.optim.AdamW(lr=p["lr"], decoupled_weight_decay=p["wd"])
        model = (
            DeepHitSingle(net, optimizer, duration_index=labtrans.cuts)
            if model_type == "deephit"
            else CoxPH(net, optimizer)
        )

        model.optimizer.set_lr(p["lr"])
        # `x_hyper` early-stops and `x_val` selects, which is the reverse of the FM
        # probe's naming. Immaterial: both are disjoint random 10% slices of train of
        # equal size, so the two conventions are the same experiment.
        model.fit(
            x_train, y_train_norm, p["batch"], epochs,
            # patience 5 rather than the torchtuples default of 10. At this width the
            # validation loss turns within a couple of epochs (train 4.04 -> 2.45 while
            # val 6.76 -> 8.03 unregularised), so 10 epochs of patience buys nothing but
            # spends ~2x the wall-clock per configuration.
            callbacks=[tt.callbacks.EarlyStopping(patience=patience)],
            val_data=(x_hyper, y_hyper),
        )

        loss = model.score_in_batches(x_val, y_val)["loss"]
        if loss < best_loss:
            best_loss, best_model, best_params = loss, model, p

    if model_type == "cox":
        best_model.compute_baseline_hazards()

    # Record which architecture won. If `layers == []` is selected, the tabular
    # DeepHit is linear -- identical in form to the FM probe head -- and the
    # "is the head allowed hidden layers?" fairness question is moot for that task.
    print(f"best DeepHit config: {best_params} (val NLL {best_loss:.4f})")

    # ---- reproducibility bundle -------------------------------------------------
    # save_survival_results already writes the net (tte_model_<tau>.pt). On its own
    # that is not enough to score new data: the column subset, the standardisation
    # and the discrete-time cuts are all fitted here. Persist them together.
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        bundle = dict(artifacts or {})
        bundle.update(
            {
                "best_params": {**best_params, "layers": list(best_params["layers"])},
                "val_nll": float(best_loss),
                "labtrans_cuts": getattr(labtrans, "cuts", None),
                "num_durations": int(out_features),
                "in_features": int(in_features),
                "model_type": model_type,
                "model_name": model_name,
                "grid_mode": grid_mode,
                "patience": patience,
                "n_configs": len(grid),
                "seed": 42,
                "net_file": f"tte_model_{5 * 365}.pt",
            }
        )
        bundle_path = os.path.join(save_path, "reproducibility.pkl")
        with open(bundle_path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"saved reproducibility bundle -> {bundle_path} "
              f"(keys: {sorted(bundle.keys())})")

    surv_df = best_model.predict_surv_df(x_test)
    metrics = evaluate_survival(
        surv_df, y_train, y_test, truncation=truncation, test_order=test_order,
        save_path=save_path, model_name=model_name, model=best_model, n_jobs=n_jobs,
        best_params={**best_params, "layers": list(best_params["layers"])},
    )
    metrics["_best_params"] = [
        {**best_params, "layers": list(best_params["layers"]), "val_nll": float(best_loss)}
    ]
    return metrics


