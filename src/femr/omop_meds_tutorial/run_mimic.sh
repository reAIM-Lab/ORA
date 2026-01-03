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

# ---- job 1: data prep (background) ----
python prepare_motor/prepare_mtpp.py \
  --pretraining_data /shared/share_mala/zj2398/models/mtpp_8k/mtpp_all_shared_6100_bin_5 \
  --athena_path "" \
  --num_bins 8 \
  --num_value_bins 5 \
  --loss_type mtpp \
  --num_final_tasks 6100 \
  --num_threads 32 \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader


# PREP_PID=$!
# echo "[prepare] PID=$PREP_PID"

# ---- job 2: training (background) ----
# multi-gpu pretraining
# export CUDA_VISIBLE_DEVICES=4,5,6,7

# accelerate launch --gpu_ids 2,3 \
#   --num_processes 2 \
# python pretrain.py \
#   --pretraining_data /data2/processed_datasets/zj2398/femr/cumc/tpp  \
#   --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
#   --per_device_train_batch_size 1 \
#   --output_dir /data2/processed_datasets/zj2398/femr/cumc_0.1/mtpp_7399/transformer  \
#   --hf_run_name cumc_tpp_new_ontology \
#   --model transformer \
#   --loss_type tpp

accelerate launch \
  --num_processes 4 \
  --gpu_ids "0,1,2,3" \
  pretrain.py \
  --pretraining_data /shared/share_mala/zj2398/models/mtpp_8k/mtpp_all_shared_6100_bin_5 \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
  --output_dir /shared/share_mala/zj2398/models/mtpp_8k/mtpp_all_shared_6100_bin_5/transformer \
  --model transformer \
  --hf_run_name mimic_mtpp_shared_aligned_task_bin_5 \
  --loss_type mtpp_shared