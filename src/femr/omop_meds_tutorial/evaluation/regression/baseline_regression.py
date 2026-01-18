import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import meds_reader
from tqdm import tqdm
from collections import defaultdict
from femr.omop_meds_tutorial.evaluation.eval_utils import _metric_mean_std, _update_metric_table

cohort_dir = Path("/shared/share_mala/zj2398/mimic/task/regression/cohort/icu_stay_4h_100k/")
result_dir = Path("/shared/share_mala/zj2398/mimic/task/regression/summary_dir")
MODEL_COL = "baseline"
N_BOOT = 100

def _bootstrap_regression_metrics(y_true, y_pred, n_boot=100, seed=0):
    """
    Returns dict:
      {"rmse": (mean,std), "mae": (mean,std), "r2": (mean,std)}
    std is bootstrap std (ddof=1). If not enough samples, returns (None,None).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n < 2:
        return {"rmse": (None, None), "mae": (None, None), "r2": (None, None)}

    rng = np.random.default_rng(seed)
    rmses, maes, r2s = [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        yt = y_true[idx]
        yp = y_pred[idx]

        rmse_b = float(np.sqrt(mean_squared_error(yt, yp)))
        mae_b = float(mean_absolute_error(yt, yp))
        r2_b = float(r2_score(yt, yp))

        rmses.append(rmse_b)
        maes.append(mae_b)
        r2s.append(r2_b)

    def _mean_std(arr):
        arr = np.asarray(arr, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        return (mean, std)

    return {"rmse": _mean_std(rmses), "mae": _mean_std(maes), "r2": _mean_std(r2s)}


for task in ["pao2","platelets","creatinine","bilirubin"]:
# for task in ["bilirubin"]:
    cohort_path =  cohort_dir/task/f"{task}.parquet"
    cohort_df = pd.read_parquet(cohort_path)
    # split_path = "/shared/share_mala/zj2398/mimic/mimic-3.1-meds/MEDS_cohort/metadata/subject_splits.parquet"
    # split = pd.read_parquet(split_path)
    # merged = cohort_df.merge(split,on="subject_id",how="inner")

    # held_out_df = merged[merged["split"] == "held_out"]
    # train_df = merged[merged["split"] != "held_out"]

    split_path = "/user/zj2398/cache/mtpp_8k/main_split.csv"
    split = pd.read_csv(split_path)
    split = split.rename(columns = {"split_name":"split"})
    merged = cohort_df.merge(split,on="subject_id",how="inner")
    test_df = merged[merged["split"] == "test"]
    train_df = merged[merged["split"] != "test"]

    print(len(train_df))
    print(len(test_df))

    meds_path = "/shared/share_mala/zj2398/mimic/data/meds_v0.6_reader"
    # Step 1: Calculate fallback values (mean per code in train_df)
    print("Calculating code means from training data...")
    code_means = train_df.groupby('code')['numerical_value'].mean().to_dict()
    print(f"Calculated means for {len(code_means)} unique codes")

    # Step 2: Load MEDS database
    print(f"Loading MEDS database from {meds_path}...")
    database = meds_reader.SubjectDatabase(meds_path)

    # Step 3: Organize test data by subject_id for efficient processing
    print("Organizing test data by subject...")
    test_by_subject = defaultdict(list)
    for idx, row in test_df.iterrows():
        test_by_subject[row['subject_id']].append({
            'index': idx,
            'prediction_time': row['prediction_time'],
            'code': row['code'],
            'true_value': row['numerical_value']
        })

    # Step 4: Define function to process each subject
    def process_subject(subject_id):
        """
        Process all test rows for a single subject.
        Returns list of (index, prediction) tuples.
        """
        if subject_id not in test_by_subject:
            return []
        
        try:
            subject = database[subject_id]
        except KeyError:
            # Subject not in database, use means for all predictions
            results = []
            for test_row in test_by_subject[subject_id]:
                pred = code_means.get(test_row['code'], np.nan)
                results.append((test_row['index'], pred))
            return results
        
        # Build a dictionary of historical values for each code
        # Key: code, Value: list of (time, value) tuples
        historical_values = defaultdict(list)
        
        for event in subject.events:
            # Store all events with their times and values
            if event.time is not None:
                # Check if event has numerical_value attribute
                if hasattr(event, 'numeric_value') and event.numeric_value is not None:
                    historical_values[event.code].append((event.time, event.numeric_value))
        
        # Process each test row for this subject
        results = []
        n_last_value_used = 0
        n_mean_used = 0
        for test_row in test_by_subject[subject_id]:
            # print(test_row)
            code = test_row['code']
            # print(f"code is {test_row['code']}, for subject {subject_id}, the code value is {historical_values[code]}")
            prediction_time = test_row['prediction_time']
            
            # Find the most recent value for this code before prediction_time
            last_value = None
            
            if code in historical_values:
                # Events are already sorted by time, so we iterate through them
                for event_time, event_value in historical_values[code]:
                    if event_time < prediction_time:
                        last_value = event_value
                    else:
                        # Since sorted, we can break early
                        break
            
            # Use last_value if found, otherwise use mean
            if last_value is not None:
                pred = last_value
                n_last_value_used += 1
            else:
                pred = code_means.get(code, np.nan)
                n_mean_used += 1
            
            results.append((test_row['index'], pred))
        
        return results,n_last_value_used, n_mean_used

    # Step 5: Process all subjects (with optional parallelization)
    print("Generating naive predictions...")
    all_predictions = {}

    total_last_value = 0
    total_mean = 0
    # Get unique subject_ids from test data
    unique_subjects = list(test_by_subject.keys())

    for subject_id in tqdm(unique_subjects):
        subject_results,n_lv, n_mean = process_subject(subject_id)
        total_last_value += n_lv
        total_mean += n_mean
        for idx, pred in subject_results:
            all_predictions[idx] = pred

    print(f"Used last value: {total_last_value}")
    print(f"Used mean fallback: {total_mean}")
    # Step 6: Add predictions to test_df
    test_df['prediction'] = test_df.index.map(all_predictions)

    # Step 7: Calculate evaluation metrics
    print("\nCalculating metrics...")
    valid_mask = ~test_df['prediction'].isna()
    n_valid = valid_mask.sum()
    n_invalid = (~valid_mask).sum()

    print(f"Valid predictions: {n_valid} ({100*n_valid/len(test_df):.2f}%)")
    print(f"Invalid predictions: {n_invalid} ({100*n_invalid/len(test_df):.2f}%)")

    # if n_valid > 0:
    #     rmse = float(np.sqrt(mean_squared_error(
    #         test_df.loc[valid_mask, "numerical_value"], 
    #         test_df.loc[valid_mask, 'prediction']
    #     )))
    #     mae = float(mean_absolute_error(
    #         test_df.loc[valid_mask, "numerical_value"], 
    #         test_df.loc[valid_mask, 'prediction']
    #     ))

    #     r2 = float(r2_score(
    #         test_df.loc[valid_mask, "numerical_value"], 
    #         test_df.loc[valid_mask, 'prediction']
    #     ))
        
    #     print(f"\n{'='*50}")
    #     print(f"Naive Baseline Results:")
    #     print(f"{'='*50}")
    #     print(f"RMSE: {rmse:.4f}")
    #     print(f"MAE:  {mae:.4f}")
    #     print(f"r2: {r2:.4f}")
    #     print(f"{'='*50}")
        
    if n_valid > 0:
        y_true = test_df.loc[valid_mask, "numerical_value"].to_numpy()
        y_pred = test_df.loc[valid_mask, "prediction"].to_numpy()

        metrics = _bootstrap_regression_metrics(
            y_true, y_pred,
            n_boot=N_BOOT,
            seed=0  # you can vary per task if you want, e.g. hash(task) & 0xffffffff
        )

        rmse_mean, rmse_std = _metric_mean_std(metrics, "rmse")
        mae_mean, mae_std   = _metric_mean_std(metrics, "mae")
        r2_mean, r2_std     = _metric_mean_std(metrics, "r2")

        print(f"\n{'='*50}")
        print(f"Naive Baseline Results (bootstrap {N_BOOT}x):")
        print(f"{'='*50}")
        print(f"RMSE: {rmse_mean:.4f} ({rmse_std:.4f})")
        print(f"MAE:  {mae_mean:.4f} ({mae_std:.4f})")
        print(f"R2:   {r2_mean:.4f} ({r2_std:.4f})")
        print(f"{'='*50}")

        _update_metric_table(result_dir, "rmse", task, MODEL_COL, rmse_mean, rmse_std)
        _update_metric_table(result_dir, "mae",  task, MODEL_COL, mae_mean,  mae_std)
        _update_metric_table(result_dir, "r2",   task, MODEL_COL, r2_mean,   r2_std)
    else:
        print("No valid predictions to evaluate!")


# Optional: Save results
# test_df.to_csv("naive_baseline_results.csv", index=False)