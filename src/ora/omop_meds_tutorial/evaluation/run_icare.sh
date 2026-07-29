bash run.sh \
  --meds_reader        /path/to/institutional_data/post_transform_meds_reader \
  --num_proc           32 \
  --model_path         /path/to/workspace/ora/institutional/icarefm/transformer/best_922142 \
  --model_name         transformer_icarefm \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --use_ntp            yes \
  --cohort_type        outcome \
  --cache_path         /path/to/hf_cache \
  --min_subjects_per_batch 8 \
  --ontology_path       /path/to/workspace/ora/institutional/survivehr/ontology.pkl \
  --main_split_path     /path/to/workspace/ora/institutional/survivehr/main_split.csv \
  --output_dir   /path/to/results/ora/institutional/outcome/  \
  --classification_cohort /path/to/institutional_data/task_labels/patient_outcomes_clean/ \
  death

bash regression/run_regression.sh \
  --meds_reader         /path/to/institutional_data/post_transform_meds_reader \
  --num_proc           24 \
  --model_path         /path/to/workspace/ora/institutional/icarefm/transformer/best_922142 \
  --model_name         transformer_icarefm \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --min_subjects_per_batch 8 \
  --ontology_path       /path/to/workspace/ora/institutional/survivehr/ontology.pkl \
  --main_split_path     /path/to/workspace/ora/institutional/survivehr/main_split.csv \
  --cohort_dir   /path/to/workspace/institutional_data/regression_100k/ \
  --output_dir   /path/to/results/ora/institutional/regression \
  --use_ntp yes \
  --cache_path  /path/to/hf_cache \
  --tasks "platelets,creatinine,spo2,glucose" \
  --regression 

bash run.sh \
  --meds_reader        /path/to/institutional_data/post_transform_meds_reader \
  --num_proc           24 \
  --model_path         /path/to/workspace/ora/institutional/icarefm/transformer/best_922142 \
  --model_name         transformer_icarefm \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --use_ntp            yes \
  --cohort_type        phenotype \
  --cache_path        /path/to/hf_cache/ \
  --min_subjects_per_batch 8 \
  --truncation  365 \
  --observation_window 730 \
  --ontology_path       /path/to/workspace/ora/institutional/survivehr/ontology.pkl \
  --main_split_path     /path/to/workspace/ora/institutional/survivehr/main_split.csv \
  --output_dir   /path/to/results/ora/institutional/phenotype  \
  --classification_cohort  /path/to/institutional_data/task_labels/phenotype_tte/binary/ \
  --tte_cohort  /path/to/institutional_data/task_labels/phenotype_tte/tte/ 