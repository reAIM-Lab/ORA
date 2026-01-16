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

# kuvira
python prepare.py \
  --pretraining_data /data2/processed_datasets/zj2398/femr/mimic/ntp \
  --ontology_path  /data2/processed_datasets/zj2398/femr/mimic/ntp/ontology.pkl \
  --loss_type ntp \
  --num_threads 32 \
  --model mamba \
  --hf_cache_path /data/huggingface/zj2398 \
  --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader

# python pretrain.py \
#   --pretraining_data /data2/processed_datasets/zj2398/femr/cumc/ntp \
#   --meds_reader /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader \
#   --per_device_train_batch_size 1 \
#   --output_dir /data2/processed_datasets/zj2398/femr/cumc/ntp/transformer \
#   --model transformer \
#   --hf_run_name mimic_transformer_ntp \
#   --loss_type ntp


# gsb
# python prepare.py \
#   --pretraining_data /user/zj2398/cache/ntp \
#   --ontology_path  /user/zj2398/cache/ntp/ontology.pkl \
#   --loss_type ntp \
#   --num_threads 32 \
#   --hf_cache_path /shared/share_mala/zj2398/huggingface/datasets \
#   --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader


# multi-gpu pretraining
# export CUDA_VISIBLE_DEVICES=0,1
accelerate launch \
  --num_processes 2 \
  pretrain.py \
  --pretraining_data /data2/processed_datasets/zj2398/femr/mimic/ntp \
  --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
  --per_device_train_batch_size 1 \
  --checkpoint_dir /data2/processed_datasets/zj2398/femr/mimic/ntp/mamba/checkpoint-151660 \
  --output_dir /data2/processed_datasets/zj2398/femr/mimic/ntp/mamba \
  --model mamba \
  --n_layers 28 \
  --hf_run_name mimic_mamba_ntp_new \
  --loss_type ntp




# accelerate launch \
#   --num_processes 2 \
#   pretrain.py \
#   --pretraining_data /user/zj2398/cache/ntp \
#   --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
#   --per_device_train_batch_size 1 \
#   --output_dir /shared/share_mala/zj2398/models/ntp_8k/transformer \
#   --model transformer \
#   # --n_layers 28 \
#   --hf_run_name mimic_transformer_ntp_shared_6100 \
#   --loss_type ntp



