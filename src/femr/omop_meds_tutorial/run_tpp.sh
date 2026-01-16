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
  --ontology_path  /user/zj2398/cache/deephit_tpp/ontology.pkl \
  --loss_type tpp \
  --tokens_per_batch 8192 \
  --model mamba \
  --num_threads 40 \
  --hf_cache_path /shared/share_mala/zj2398/huggingface/datasets \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader
  
  # >"$PREP_LOG" 2>&1 &

# ---- job 2: training (background) ----
# multi-gpu pretraining
# export CUDA_VISIBLE_DEVICES=0,1,2,3


accelerate launch \
  --num_processes 4 \
  pretrain.py \
  --pretraining_data /user/zj2398/cache/deephit_tpp \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
  --per_device_train_batch_size 1 \
  --output_dir /shared/share_mala/zj2398/models/tpp_8k/mamba_2e-4_5.0 \
  --learning_rate 2e-4 \
  --max_grad_norm 5.0 \
  --model mamba \
  --n_layers 28 \
  --hf_run_name mimic_mamba_tpp_2e-4_5.0 \
  --loss_type tpp
