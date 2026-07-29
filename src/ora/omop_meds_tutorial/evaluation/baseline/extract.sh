conda activate ora

# CUIMC
## TTE
export TASK_DIR=/path/to/institutional_data/task_labels/phenotype_tte/tte/AMI/
export OMOP_MEDS_READER=/path/to/institutional_data/post_transform_meds_reader/
export PRETRAINING_DATA=/path/to/institutional_data/baseline_features_22_12_vj2292/AMI/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"

export TASK_DIR=/path/to/institutional_data/task_labels/phenotype_tte/tte/Ischemic_Stroke/
export OMOP_MEDS_READER=/path/to/institutional_data/post_transform_meds_reader/
export PRETRAINING_DATA=/path/to/institutional_data/baseline_features_22_12_vj2292/Ischemic_Stroke/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"

export TASK_DIR=/path/to/institutional_data/task_labels/phenotype_tte/tte/MASLD/
export OMOP_MEDS_READER=/path/to/institutional_data/post_transform_meds_reader/
export PRETRAINING_DATA=/path/to/institutional_data/baseline_features_22_12_vj2292/MASLD/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"

export TASK_DIR=/path/to/institutional_data/task_labels/phenotype_tte/tte/Celiac/
export OMOP_MEDS_READER=/path/to/institutional_data/post_transform_meds_reader/
export PRETRAINING_DATA=/path/to/institutional_data/baseline_features_22_12_vj2292/Celiac/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"


## Classification
export TASK_DIR=/path/to/institutional_data/task_labels/patient_outcomes_clean/death
export OMOP_MEDS_READER=/path/to/institutional_data/post_transform_meds_reader/
export PRETRAINING_DATA=/path/to/institutional_data/baseline_features_22_12_vj2292/death_binary/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"

export TASK_DIR=/path/to/institutional_data/task_labels/patient_outcomes_clean/long_los/
export OMOP_MEDS_READER=/path/to/institutional_data/post_transform_meds_reader/
export PRETRAINING_DATA=/path/to/institutional_data/baseline_features_22_12_vj2292/long_los_binary/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"

export TASK_DIR=/path/to/institutional_data/task_labels/patient_outcomes_clean/readmission/
export OMOP_MEDS_READER=/path/to/institutional_data/post_transform_meds_reader/
export PRETRAINING_DATA=/path/to/institutional_data/baseline_features_22_12_vj2292/readmission_binary/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"


export TASK_DIR=/path/to/institutional_data/task_labels/phenotype_tte/binary/AMI/
export OMOP_MEDS_READER=/path/to/institutional_data/post_transform_meds_reader/
export PRETRAINING_DATA=/path/to/institutional_data/baseline_features_22_12_vj2292/AMI_binary/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"

export TASK_DIR=/path/to/institutional_data/task_labels/phenotype_tte/binary/Ischemic_Stroke/
export OMOP_MEDS_READER=/path/to/institutional_data/post_transform_meds_reader/
export PRETRAINING_DATA=/path/to/institutional_data/baseline_features_22_12_vj2292/Ischemic_Stroke_binary/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"

export TASK_DIR=/path/to/institutional_data/task_labels/phenotype_tte/binary/MASLD/
export OMOP_MEDS_READER=/path/to/institutional_data/post_transform_meds_reader/
export PRETRAINING_DATA=/path/to/institutional_data/baseline_features_22_12_vj2292/MASLD_binary/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"


export TASK_DIR=/path/to/institutional_data/task_labels/phenotype_tte/binary/Celiac/
export OMOP_MEDS_READER=/path/to/institutional_data/post_transform_meds_reader/
export PRETRAINING_DATA=/path/to/institutional_data/baseline_features_22_12_vj2292/Celiac_binary/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"


conda activate ora

# MIMIC
## TTE
export TASK_DIR=/path/to/mimic_data/phenotype_task/tte/ami/
export OMOP_MEDS_READER=/path/to/mimic_meds_reader/
export PRETRAINING_DATA=/path/to/mimic_data/baseline_features_22_12_vj2292/ami/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"

export TASK_DIR=/path/to/mimic_data/phenotype_task/tte/stroke/
export OMOP_MEDS_READER=/path/to/mimic_meds_reader/
export PRETRAINING_DATA=/path/to/mimic_data/baseline_features_22_12_vj2292/stroke/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"

export TASK_DIR=/path/to/mimic_data/phenotype_task/tte/masld/
export OMOP_MEDS_READER=/path/to/mimic_meds_reader/
export PRETRAINING_DATA=/path/to/mimic_data/baseline_features_22_12_vj2292/masld/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"




## Classification
export TASK_DIR=/path/to/mimic_data/patient_outcome_tasks/in_hospital_mortality/
export OMOP_MEDS_READER=/path/to/mimic_meds_reader/
export PRETRAINING_DATA=/path/to/mimic_data/baseline_features_22_12_vj2292/in_hospital_mortality_binary/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"

export TASK_DIR=/path/to/mimic_data/patient_outcome_tasks/long_los/
export OMOP_MEDS_READER=/path/to/mimic_meds_reader/
export PRETRAINING_DATA=/path/to/mimic_data/baseline_features_22_12_vj2292/long_los_binary/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"

export TASK_DIR=/path/to/mimic_data/patient_outcome_tasks/readmission/
export OMOP_MEDS_READER=/path/to/mimic_meds_reader/
export PRETRAINING_DATA=/path/to/mimic_data/baseline_features_22_12_vj2292/readmission_binary/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"


export TASK_DIR=/path/to/mimic_data/phenotype_task/binary/ami/
export OMOP_MEDS_READER=/path/to/mimic_meds_reader/
export PRETRAINING_DATA=/path/to/mimic_data/baseline_features_22_12_vj2292/ami_binary/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"

export TASK_DIR=/path/to/mimic_data/phenotype_task/binary/stroke/
export OMOP_MEDS_READER=/path/to/mimic_meds_reader/
export PRETRAINING_DATA=/path/to/mimic_data/baseline_features_22_12_vj2292/stroke_binary/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"

export TASK_DIR=/path/to/mimic_data/phenotype_task/binary/masld/
export OMOP_MEDS_READER=/path/to/mimic_meds_reader/
export PRETRAINING_DATA=/path/to/mimic_data/baseline_features_22_12_vj2292/masld_binary/

python -u -m ora.omop_meds_tutorial.generate_tabular_features --pretraining_data "$PRETRAINING_DATA" --meds_reader "$OMOP_MEDS_READER" --cohort_dir "$TASK_DIR"