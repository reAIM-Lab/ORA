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
python prepare_motor/prepare_motor.py \
    --pretraining_data /data/models/zj2398/femr/cumc_0.1/mtpp_5_bin \
    --athena_path "" \
    --num_bins 8 \
    --num_value_bins 5 \
    --loss_type mtpp \
    --num_threads 16 \
    --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_0.1_reader 
  >"$PREP_LOG" 2>&1 &

# PREP_PID=$!
# echo "[prepare] PID=$PREP_PID"

# ---- job 2: training (background) ----
# multi-gpu pretraining
# CUDA_VISIBLE_DEVICES=2 accelerate launch \
#   --num_processes 2 \
#   --mixed_precision bf16 \
#   --gpu_ids "0,2" \
#   pretrain_motor.py \
#   --pretraining_data /data/processed_datasets/processed_datasets/zj2398/femr/mimic/deephit_tpp \
#   --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
#   --per_device_train_batch_size 1 \
#   --output_dir /data/processed_datasets/processed_datasets/zj2398/femr/mimic/deephit_tpp/output_mamba \
#   --model mamba \
#   --loss_type tpp 

#single gpu
python pretrain.py \
  --pretraining_data /data/models/zj2398/femr/cumc_0.1/mtpp_5_bin \
  --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_0.1_reader \
  --per_device_train_batch_size 1 \
  --output_dir /data/models/zj2398/femr/cumc_0.1/mtpp_5_bin/transformer \
  --model transformer \
  --hf_run_name mtpp_bin_5_cumc_0.1 \
  --loss_type mtpp \


#  825, 8, 6100] torch.Size([825, 8, 6100])