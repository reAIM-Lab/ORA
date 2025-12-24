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
# python prepare_motor/prepare_mtpp.py \
#     --pretraining_data /user/zj2398/cache/mtpp_8k_num_bin5 \
#     --athena_path "" \
#     --num_bins 8 \
#     --num_value_bins 5 \
#     --value_count_threshold 1000 \
#     --loss_type mtpp \
#     --num_threads 32 \
#     --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader
#   >"$PREP_LOG" 2>&1 &

python prepare_motor/prepare_mtpp.py \
    --pretraining_data /data2/processed_datasets/zj2398/femr/cumc_0.1/mtpp \
    --athena_path "" \
    --num_bins 8 \
    --value_count_threshold 100 \
    --loss_type mtpp \
    --num_threads 24  \
    --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_0.1_reader
  >"$PREP_LOG" 2>&1 &

# PREP_PID=$!
# echo "[prepare] PID=$PREP_PID"

# ---- job 2: training (background) ----
# multi-gpu pretraining
# export CUDA_VISIBLE_DEVICES=2

# accelerate launch --gpu_ids 2,3 \
#   --num_processes 2 \
python pretrain.py \
  --pretraining_data /data2/processed_datasets/zj2398/femr/cumc_0.1/mtpp  \
  --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_0.1_reader \
  --per_device_train_batch_size 1 \
  --output_dir /data2/processed_datasets/zj2398/femr/cumc_0.1/mtpp/transformer  \
  --hf_run_name cumc_0.1_new_ontology_mtpp \
  --model transformer \
  --loss_type mtpp_shared

#single gpu
# python pretrain.py \
#     --pretraining_data /data/models/zj2398/femr/cumc_0.1/mtpp_shared \
#     --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_0.1_reader \
#     --output_dir /data/models/zj2398/femr/cumc_full/tpp/transformer \
#     --model transformer \
#     --hf_run_name cumc_full_tpp \
#     --loss_type tpp
