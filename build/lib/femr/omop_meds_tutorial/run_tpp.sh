#!/usr/bin/env bash
set -euo pipefail

# ---- config (edit if you like) ----
# LOG_DIR="${LOG_DIR:-./logs}"
# PREP_LOG="$LOG_DIR/prepare_motor_$(date +%Y%m%d_%H%M%S).log"
# TRAIN_LOG="$LOG_DIR/train_mamba_$(date +%Y%m%d_%H%M%S).log"
# mkdir -p "$LOG_DIR"

echo "[$(date)] starting both jobs..."
# echo "Logs:"
# echo "  prepare -> $PREP_LOG"
# echo "  train   -> $TRAIN_LOG"
# echo

python prepare.py \
  --pretraining_data /user/zj2398/cache/deephit_tpp \
  --ontology_path  /user/zj2398/cache/mtpp_8k_num_bin5/ontology.pkl \
  --time_bins 8 \
  --loss_type tpp \
  --num_threads 32 \
  --hf_cache_path /shared/share_mala/zj2398/huggingface/datasets \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader
  # >"$PREP_LOG" 2>&1 &

# ---- job 2: training (background) ----
# multi-gpu pretraining
# export CUDA_VISIBLE_DEVICES=0,3,5


accelerate launch \
  --num_processes 3 \
  pretrain.py \
  --pretraining_data /user/zj2398/cache/deephit_tpp \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
  --per_device_train_batch_size 1 \
  --output_dir /shared/share_mala/zj2398/models/tpp_8k/mamba \
  --model mamba \
  --n_layers 28 \
  --hf_run_name mimic_mamba_tpp \
  --loss_type tpp
