import os
import pandas as pd
import numpy as np

# Acute phenotypes can recur; labeling uses the first event strictly in the future.
ACUTE_PHENOTYPES = {"AMI", "Ischemic_Stroke"}

'''
cohort_start date case
'''


def robust_to_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed")  # pandas >= 2.0
    except TypeError:
        return pd.to_datetime(s, errors="coerce")  # older pandas fallback


def min_datetime(a: pd.Series, b: pd.Series) -> pd.Series:
    if a is None and b is None:
        return pd.Series(dtype="datetime64[ns]")
    if a is None:
        a = pd.Series(pd.NaT, index=b.index, dtype="datetime64[ns]")
    if b is None:
        b = pd.Series(pd.NaT, index=a.index, dtype="datetime64[ns]")
    a = a.astype("datetime64[ns]")
    b = b.astype("datetime64[ns]")
    return pd.concat([a, b], axis=1).min(axis=1)

# Ensures each patient appears once in phenotype data before merging.
def dedupe_cohort_cases(pheno_cases: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure one row per subject_id for the right side of the merge.
    Aggregate to the earliest cohort_start_date and the latest cohort_end_date.
    """
    cols = []
    if "cohort_start_date" in pheno_cases.columns:
        pheno_cases["cohort_start_date"] = robust_to_datetime(pheno_cases["cohort_start_date"]).astype("datetime64[ns]")
        cols.append("cohort_start_date")
    if "cohort_end_date" in pheno_cases.columns:
        pheno_cases["cohort_end_date"] = robust_to_datetime(pheno_cases["cohort_end_date"]).astype("datetime64[ns]")
        cols.append("cohort_end_date")

    if not cols:
        return pheno_cases.drop_duplicates(subset=["subject_id"], keep="first")

    agg_spec = {}
    if "cohort_start_date" in cols:
        agg_spec["cohort_start_date"] = "min"
    if "cohort_end_date" in cols:
        agg_spec["cohort_end_date"] = "max"

    deduped = (
        pheno_cases.groupby("subject_id", as_index=False, sort=False)[list(agg_spec.keys())]
        .agg(agg_spec)
        .astype({k: "datetime64[ns]" for k in agg_spec})
    )
    return deduped


'''
cases time, censor time, prediction time
if cases exsit and cases time > prediction time -> yes
else:  it is censor
'''

def generate_time_to_event_labels(
    phenotype_folder: str,
    prediction_df: pd.DataFrame,
    cohort_info_df: pd.DataFrame,
    last_event_df: pd.DataFrame | None = None,
    fail_fast: bool = False,
) -> pd.DataFrame:
    # filter out rows that belong to a disease
    pheno_cases_raw = cohort_info_df[cohort_info_df["phenotype_folder"] == phenotype_folder].copy()

    prediction_df = prediction_df.copy()
    if "subject_id" not in prediction_df.columns:
        raise ValueError("prediction_df must contain 'subject_id'")
    if "prediction_time" not in prediction_df.columns:
        raise ValueError("prediction_df must contain 'prediction_time'")
    prediction_df["prediction_time"] = robust_to_datetime(prediction_df["prediction_time"]).astype("datetime64[ns]")

    # Ensures each patient appears once in phenotype data before merging.
    pheno_cases = dedupe_cohort_cases(pheno_cases_raw)

    merged = prediction_df.merge(
        pheno_cases,
        on="subject_id",
        how="left",
        validate="many_to_one",  # now guaranteed by dedupe
    )

    if last_event_df is not None and not last_event_df.empty:
        tmp = last_event_df.copy()
        if "subject_id" not in tmp.columns:
            raise ValueError("last_event_df must have a 'subject_id' column")
        if "last_event_time" not in tmp.columns:
            raise ValueError("last_event_df must have a 'last_event_time' column")
        tmp["last_event_time"] = robust_to_datetime(tmp["last_event_time"]).astype("datetime64[ns]")
        tmp = tmp.dropna(subset=["last_event_time"])
        tmp = tmp.groupby("subject_id", as_index=False, sort=False)["last_event_time"].max()
        merged = merged.merge(
            tmp[["subject_id", "last_event_time"]],
            on="subject_id",
            how="left",
            validate="many_to_one",
        )
    else:
        merged["last_event_time"] = pd.NaT
        merged["last_event_time"] = merged["last_event_time"].astype("datetime64[ns]")

    merged["censor_time"] = min_datetime(merged.get("cohort_end_date"), merged.get("last_event_time")).astype(
        "datetime64[ns]"
    )

    #  have case event and it happens in the future
    occurs_in_future = merged["cohort_start_date"].notna() & (
        merged["cohort_start_date"] >= merged["prediction_time"]
    )

    merged["event_time_candidate"] = merged["cohort_start_date"].where(occurs_in_future).astype("datetime64[ns]")

    within_censor = merged["censor_time"].isna() | (merged["event_time_candidate"] <= merged["censor_time"])
    merged["event_observed"] = merged["event_time_candidate"].notna() & within_censor

    fallback_censor = merged["censor_time"].combine_first(merged.get("cohort_end_date")).fillna(
        merged["prediction_time"]
    )
    fallback_censor = fallback_censor.astype("datetime64[ns]")

    merged["target_time"] = merged["event_time_candidate"].where(merged["event_observed"], fallback_censor).astype(
        "datetime64[ns]"
    )

    merged["time_to_event_hours"] = (merged["target_time"] - merged["prediction_time"]).dt.total_seconds() / 3600.0

    negatives = merged["time_to_event_hours"] < 0
    if negatives.any():
        merged.loc[negatives, "event_observed"] = False
        merged.loc[negatives, "target_time"] = merged.loc[negatives, "prediction_time"]
        merged.loc[negatives, "time_to_event_hours"] = 0.0

    missing_tte = merged["time_to_event_hours"].isna()
    if missing_tte.any():
        merged.loc[missing_tte, "event_observed"] = False
        merged.loc[missing_tte, "target_time"] = merged.loc[missing_tte, "prediction_time"]
        merged.loc[missing_tte, "time_to_event_hours"] = 0.0

    if fail_fast:
        bad = merged[(merged["time_to_event_hours"] < 0) | (merged["time_to_event_hours"].isna())]
        if len(bad) > 0:
            raise ValueError(
                "Labeling produced negative/NaN TTE after clamping; investigate inputs. Offending rows:\n" + str(bad)
            )

    return merged.drop(
        columns=[
            "cohort_start_date",
            "cohort_end_date",
            "last_event_time",
            "event_time_candidate",
            "target_time",
            "censor_time",
        ],
        errors="ignore",
    )


if __name__ == "__main__":
    cases_csv_path = os.environ.get("TTE_CASES_CSV", "yl_pheno_cases.csv")
    data_path = os.environ.get(
        "TTE_DATA_DIR",
        "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/task_labels/phenotype_sample",
    )
    last_event_path = os.environ.get("TTE_LAST_EVENT_CSV", "last_event_times.csv")
    output_dir = os.environ.get(
        "TTE_OUTPUT_DIR",
        "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/task_labels/tte_task_labels",
    )

    if not os.path.exists(cases_csv_path):
        raise FileNotFoundError(f"The file '{cases_csv_path}' was not found.")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The folder '{data_path}' was not found.")

    cases_df = pd.read_csv(cases_csv_path)
    if "phenotype_folder" not in cases_df.columns and "cohort_name" in cases_df.columns:
        # add a new column that has Celiac Cases -> Celiac
        cases_df["phenotype_folder"] = (
            cases_df["cohort_name"].str.replace(" Cases", "", regex=False).str.replace(" ", "_", regex=False)
        )
    # Normalize once; dedupe happens per-phenotype inside generate_time_to_event_labels
    for col in ["cohort_start_date", "cohort_end_date"]:
        if col in cases_df.columns:
            cases_df[col] = robust_to_datetime(cases_df[col]).astype("datetime64[ns]")

    if os.path.exists(last_event_path):
        last_event_df = pd.read_csv(last_event_path)
        if "subject_id" not in last_event_df.columns:
            raise ValueError("last_event_times.csv must contain a 'subject_id' column")
        if "last_event_time" not in last_event_df.columns:
            raise ValueError("last_event_times.csv must contain a 'last_event_time' column")
        last_event_df["last_event_time"] = robust_to_datetime(last_event_df["last_event_time"]).astype("datetime64[ns]")
        last_event_df = last_event_df.dropna(subset=["last_event_time"])
        last_event_df = last_event_df.groupby("subject_id", as_index=False, sort=False)["last_event_time"].max()
    else:
        last_event_df = pd.DataFrame(columns=["subject_id", "last_event_time"])

    phenotype_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    splits = ["train", "tuning", "held_out"]

    for phenotype in phenotype_folders:
        pheno_dir = os.path.join(data_path, phenotype)
        out_dir = os.path.join(output_dir, phenotype)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n--- Processing phenotype: {phenotype} ---")

        for split in splits:
            file_path = os.path.join(pheno_dir, f"{split}.parquet")
            if not os.path.exists(file_path):
                print(f"Skipping {file_path} - file not found.")
                continue

            print(f"  - Processing split: {split}")
            pred_df = pd.read_parquet(file_path)

            labeled = generate_time_to_event_labels(
                phenotype_folder=phenotype,
                prediction_df=pred_df,
                cohort_info_df=cases_df,
                last_event_df=last_event_df,
                fail_fast=False,
            )

            output_path = os.path.join(out_dir, f"tte_{split}.parquet")
            labeled.to_parquet(output_path)
            print(f"    - Saved results to {output_path}")
