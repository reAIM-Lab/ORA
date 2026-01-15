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
  --pretraining_data /user/zj2398/cache/mtpp_all_shared_6100_bin_5 \
  --ontology_path /user/zj2398/cache/mtpp_8k_num_bin5/ontology.pkl \
  --time_bins 8 \
  --num_threads 64 \
  --value_bins 5 \
  --num_final_tasks 6100 \
  --tokens_per_batch 16384 \
  --model mamba \
  --loss_type mtpp \
  --hf_cache_path /shared/share_mala/zj2398/huggingface/datasets \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader
  # >"$PREP_LOG" 2>&1 &

# ---- job 2: training (background) ----
# multi-gpu pretraining
# export CUDA_VISIBLE_DEVICES=4,5,6,7

# accelerate launch \
#   --num_processes 4 \
#   pretrain.py \
#     --pretraining_data  /user/zj2398/cache/mtpp_all_shared_6100_bin_5 \
#     --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
#     --per_device_train_batch_size 1 \
#     --learning_rate 5e-6 \
#     --output_dir /shared/share_mala/zj2398/models/mtpp_8k/mtpp_all_shared_6100_bin_5/mamba  \
#     --hf_run_name mimic_mamba_revised_mtpp_6100_grad_clip \
#     --model mamba \
#     --n_layers 28 \
#     --loss_type mtpp_shared

  accelerate launch \
    --num_processes 6 \
    pretrain.py \
    --pretraining_data  /user/zj2398/cache/mtpp_all_shared_6100_bin_5 \
    --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --output_dir /shared/share_mala/zj2398/models/mtpp_8k/mtpp_all_shared_6100_bin_5/mamba  \
    --hf_run_name mimic_mamba_revised_mtpp_6100_1e-5 \
    --model mamba \
    --n_layers 28 \
    --loss_type mtpp_shared


  # python pretrain.py  \
  # --pretraining_data /user/zj2398/cache/mtpp_all_shared_6100_bin_5   \
  # --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader  \
  # --per_device_train_batch_size 1   \
  # --output_dir /user/zj2398/cache/mtpp_all_shared_6100_bin_5/mamba \
  # --model mamba  \
  # --n_layers 28  \
  # --hf_run_name mimic_mamba_mtpp_shared_6100_bin_5  \
  # --loss_type mtpp_shared