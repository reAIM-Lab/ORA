#!/usr/bin/env bash
set -euo pipefail

# ---- config (edit if you like) ----
LOG_DIR="${LOG_DIR:-./logs}"
PREP_LOG="$LOG_DIR/prepare_motor_$(date +%Y%m%d_%H%M%S).log"
TRAIN_LOG="$LOG_DIR/train_mamba_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

echo "[$(date)] starting both jobs..."
echo "Logs:"
echo "  prepare -> $PREP_LOG"
echo "  train   -> $TRAIN_LOG"
echo

python prepare.py \
  --pretraining_data /data2/processed_datasets/zj2398/femr/cumc/mtpp \
  --ontology_path  ontology_path \
  --num_bins 8 \
  --loss_type mtpp \
  --num_threads 32 \
  --num_final_tasks 6972 \
  --hf_cache_path /data2/huggingface/zj2398 \
  --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader
  # >"$PREP_LOG" 2>&1 &

# ---- job 2: training (background) ----
# multi-gpu pretraining
# export CUDA_VISIBLE_DEVICES=3 

# accelerate launch --gpu_ids 2,3 \
#   --num_processes 2 \
python pretrain.py \
  --pretraining_data /data2/processed_datasets/zj2398/femr/cumc/mtpp  \
  --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
  --per_device_train_batch_size 1 \
  --output_dir /data2/processed_datasets/zj2398/femr/cumc/mtpp/transformer  \
  --hf_run_name cumc_mtpp_new_ontology_7399 \
  --model transformer \
  --loss_type mtpp_shared

