"""
Most Recent Imputation Baseline for Regression Tasks

Strategy:
- For each test sample, find the most recent historical value of the target code 
  before the prediction time
- If no historical value exists, use the mean from training data

Uses meds_reader's parallel map/reduce for efficient processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from femr.omop_meds_tutorial.evaluation.eval_utils import _metric_mean_std, _update_metric_table,_bootstrap_regression_metrics
import meds_reader
import functools
from typing import Dict, List, Tuple
from collections import defaultdict

# ============================================================================
# Configuration
# ============================================================================
# COHORT_DIR = Path("/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/task_labels/regression/")
COHORT_DIR = Path("/data2/processed_datasets/zj2398/cumc_data/regression_100k/")
MEDS_PATH = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader"
SPLIT_PATH = "/data2/processed_datasets/zj2398/femr/cumc/motor/main_split.csv"

# OUTPUT_DIR = "/data/results/zj2398/femr/cumc_0.1/regression_2_5_months/most_recent"

CODE_DICT = {
    "platelets": "LOINC/26515-7//UCUM/10*9/L",
    "creatinine": "LOINC/2160-0//UCUM/mg/dL",
    "bilirubin": "LOINC/1968-7//UCUM/mg/dL",
    "spo2":"LOINC/2708-6//UCUM/%",
}

MODEL_COL = "Most Recent Imputation"
# ============================================================================
# Data Loading
# ============================================================================
def load_cohort_data(task: str) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Load cohort and split into train/test. Returns (train_df, test_df, train_mean)."""
    cohort_path = COHORT_DIR / task / f"{task}.parquet"
    cohort_df = pd.read_parquet(cohort_path)
    
    split = pd.read_csv(SPLIT_PATH).rename(columns={"split_name": "split"})
    merged = cohort_df.merge(split, on="subject_id", how="inner")
    
    test_df = merged[merged["split"] == "test"].reset_index(drop=True)
    train_df = merged[merged["split"] != "test"]
    train_mean = train_df['numerical_value'].mean()
    
    return train_df, test_df, train_mean


def build_test_lookup(test_df: pd.DataFrame) -> Dict[int, List[Tuple]]:
    """
    Build lookup: subject_id -> [(row_idx, prediction_time), ...]
    Using tuples instead of dicts for memory efficiency.
    """
    test_lookup = defaultdict(list)
    for idx, row in test_df.iterrows():
        test_lookup[row['subject_id']].append((idx, row['prediction_time']))
    return dict(test_lookup)


# ============================================================================
# Parallel Processing Functions
# ============================================================================
def process_subjects(
    subject_iter,
    target_code: str,
    test_lookup: Dict[int, List[Tuple]],
    train_mean: float
) -> Dict[int, float]:
    """
    Worker function for parallel processing.
    
    For each subject in the iterator:
    1. Check if subject has test samples
    2. Collect all historical values for target_code (already time-sorted)
    3. For each test sample, find most recent value before prediction_time
    
    Returns: {row_idx: predicted_value}
    """
    results = {}
    
    for subject in subject_iter:
        count = [0,0]
        sid = subject.subject_id
        
        # Skip subjects not in test set
        if sid not in test_lookup:
            continue
        
        # Collect historical values: [(time, value), ...]
        # Events are already sorted by time in meds_reader
        history = [
            (event.time, event.numeric_value)
            for event in subject.events
            if event.code == target_code 
            and event.time is not None 
            and event.numeric_value is not None and event.numeric_value != 0
        ]
        
        # Process each test sample for this subject
        for row_idx, pred_time in test_lookup[sid]:
            # Find most recent value before prediction_time
            # Since history is sorted, we iterate and keep updating

            last_val = None
            for event_time, event_val in history:
                if event_time < pred_time:
                    last_val = event_val
                else:
                    break  # Past prediction time, stop
            if last_val is not None:
                count[0] += 1
            else:
                count[1] += 1
            results[row_idx] = last_val if last_val is not None else train_mean
            # results[row_idx] = train_mean
        # print(f"subject {subject.subject_id} ratio of last event is {count[0]/(count[0]+count[1])}")
    
    return results


def aggregate_results(accumulated: Dict[int, float], chunk: Dict[int, float]) -> Dict[int, float]:
    """Merge results from parallel workers."""
    accumulated.update(chunk)
    return accumulated
    


# ============================================================================
# Main
# ============================================================================
def main():
    print("Initializing MEDS database...")
    db = meds_reader.SubjectDatabase(MEDS_PATH, num_threads=40)
    
    all_results = []
    
    for task, code in CODE_DICT.items():
        print(f"\n{'='*60}")
        print(f"Task: {task} | Code: {code}")
        print('='*60)
        
        # Load data
        train_df, test_df, train_mean = load_cohort_data(task)
        test_lookup = build_test_lookup(test_df)
        
        print(f"Train: {len(train_df):,} | Test: {len(test_df):,} | Mean: {train_mean:.4f}")
        print(f"Unique test subjects: {len(test_lookup):,}")
        
        # Create worker with bound arguments
        worker = functools.partial(
            process_subjects,
            target_code=code,
            test_lookup=test_lookup,
            train_mean=train_mean
        )
        
        # Parallel map-reduce
        print("Running parallel inference...")
        predictions = functools.reduce(
            aggregate_results,
            db.map(worker),
            {}
        )
        
        # Evaluate
        # metrics = evaluate(test_df, predictions, task)
        # all_results.append(metrics)

        y_pred = test_df.index.map(predictions)
        y_true = test_df['numerical_value']

        metrics = _bootstrap_regression_metrics(
            y_true, y_pred,
            n_boot=100,
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
    
    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

# ============================================================================
# Evaluation
# # # ============================================================================
# def evaluate(test_df: pd.DataFrame, predictions: Dict[int, float], task: str) -> Dict:
#     """Calculate and print metrics."""
#     test_df = test_df.copy()
#     test_df['prediction'] = test_df.index.map(predictions)
    
#     # valid_mask = ~test_df['prediction'].isna()
#     # n_valid = valid_mask.sum()
#     # n_total = len(test_df)
    
#     print(f"\nResults for {task}:")
#     # print(f"  Valid predictions: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    
#     # if n_valid > 0:
#     #     y_true = test_df.loc[valid_mask, 'numerical_value']
#     #     y_pred = test_df.loc[valid_mask, 'prediction']
    
#     y_true = test_df['numerical_value']
#     y_pred = test_df['prediction']
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
    
#     print(f"  RMSE: {rmse:.4f}")
#     print(f"  MAE:  {mae:.4f}")
#     print(f"  R²:   {r2:.4f}")
    
#     return {'task': task, 'rmse': rmse, 'mae': mae, 'r2': r2}