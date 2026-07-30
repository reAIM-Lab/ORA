#!/usr/bin/env bash
# Pretrain ORA (loss_type=mtpp_shared) on MIMIC-IV with the transformer backbone.
#
# This is the proposed objective: one head over the joint (time, value) table, shared across
# codes.  Its ablation is --loss_type mtpp_factorized, which forces that table to factorise
# into one distribution for *when* and one for *what*.
#
# FAIR COMPARISON: both load the same task file (mtpp_task.pkl) and the same batches, so the
# head is the only difference.  To reproduce that pairing, run this script, then re-run it with
# LOSS_TYPE=mtpp_factorized against the same PRETRAINING_DATA and a different OUTPUT_DIR --
# prepare.py skips artifacts that already exist, so nothing is rebuilt:
#
#   bash run_ora.sh
#   LOSS_TYPE=mtpp_factorized OUTPUT_DIR=$STORAGE/mtpp_factorized/transformer bash run_ora.sh
#
# Keep NUM_PROCESSES the same across the two runs: the effective batch size is
# NUM_PROCESSES x PER_DEVICE_TRAIN_BATCH_SIZE, so a different process count silently changes it.
#
# Usage:   bash run_ora.sh
# Resume:  CHECKPOINT_DIR=<path to a checkpoint-* dir> bash run_ora.sh
set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------- configuration
# Activate the `tte` environment unless one is already active.
VENV="${VENV:-/path/to/envs/tte}"
if [[ -z "${VIRTUAL_ENV:-}" && -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

STORAGE="${STORAGE:-/path/to/workspace/ora}"

# Inputs.
MEDS_READER="${MEDS_READER:-$STORAGE/meds_v0.6_reader}"
ONTOLOGY_PATH="${ONTOLOGY_PATH:-$STORAGE/metadata/ontology.pkl}"

# Outputs: tokenizer, mtpp_task.pkl, train_batches, val_batches, main_split.csv, and the model.
# mtpp_shared is ORA. Set LOSS_TYPE=mtpp_factorized to train the ablation against the
# same prepared task and batches; give it its own OUTPUT_DIR.
LOSS_TYPE="${LOSS_TYPE:-mtpp_shared}"
PRETRAINING_DATA="${PRETRAINING_DATA:-$STORAGE/mtpp_shared}"
OUTPUT_DIR="${OUTPUT_DIR:-$PRETRAINING_DATA/transformer}"
HF_CACHE_PATH="${HF_CACHE_PATH:-$STORAGE/hf_cache}"

# Discretisation and vocabulary: T=8, V=5, 6100 pretraining tasks, as reported in the paper.
# Unlike ICareFM and SurvivEHR, the mtpp family does read --num_final_tasks, --time_bins and
# --value_bins: prepare.py forwards them only to the mtpp/mtpp_shared/mtpp_value tasks.
TIME_BINS="${TIME_BINS:-8}"
VALUE_BINS="${VALUE_BINS:-5}"
NUM_FINAL_TASKS="${NUM_FINAL_TASKS:-6100}"
NUM_THREADS="${NUM_THREADS:-32}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"

# n_epochs, learning rate, early stopping and layer count are left at pretrain.py and Hydra
# defaults (20 epochs, lr 1e-5, patience 3; 11 layers / 768 hidden / 12 heads from
# conf/model/transformer.yaml), which is what the reported transformer checkpoints used.
#
# The lowest-eval_loss model is mirrored to $OUTPUT_DIR/best_model every time validation loss
# improves, and rewritten once when training ends.  Point evaluation --model_path there.

# Weights & Biases.  NOTE: pretrain.py currently sets report_to="none", so nothing is logged and
# RUN_NAME is only recorded in the HF trainer state.  Change it to report_to="wandb" to log.
RUN_NAME="${RUN_NAME:-mimic_transformer_${LOSS_TYPE}_${NUM_FINAL_TASKS}_T${TIME_BINS}_V${VALUE_BINS}_$(date +%Y%m%d_%H%M%S)}"
export WANDB_PROJECT="${WANDB_PROJECT:-huggingface}"
export WANDB_NAME="$RUN_NAME"

LOG_DIR="${LOG_DIR:-$PRETRAINING_DATA/logs}"

# ---------------------------------------------------------------- preflight
for path in "$MEDS_READER" "$ONTOLOGY_PATH" "$MEDS_READER/metadata/subject_splits.parquet"; do
  [[ -e "$path" ]] || { echo "ERROR: missing input: $path" >&2; exit 1; }
done
mkdir -p "$PRETRAINING_DATA" "$HF_CACHE_PATH" "$LOG_DIR"

# pretrain.py reads the ontology from $PRETRAINING_DATA/ontology.pkl rather than from a flag, so
# link the shared one into place instead of copying it.
[[ -e "$PRETRAINING_DATA/ontology.pkl" ]] || ln -s "$ONTOLOGY_PATH" "$PRETRAINING_DATA/ontology.pkl"

echo "=============================================================="
echo " ORA ($LOSS_TYPE) | MIMIC-IV | Transformer"
echo "   meds_reader      : $MEDS_READER"
echo "   ontology         : $ONTOLOGY_PATH"
echo "   pretraining_data : $PRETRAINING_DATA"
echo "   output_dir       : $OUTPUT_DIR"
echo "   T x V            : $TIME_BINS x $VALUE_BINS   (tasks: $NUM_FINAL_TASKS)"
echo "   GPUs             : $GPUS  ($NUM_PROCESSES processes)"
echo "   run name         : $RUN_NAME"
echo "=============================================================="

# ---------------------------------------------------------------- 1. data preparation
# prepare.py maps mtpp_shared onto the mtpp task, so this writes mtpp_task.pkl -- the same file
# the factorized ablation loads.  Existing artifacts are skipped, so re-running is safe.
echo "[$(date)] step 1/2: preparing tokenizer, task and batches"
python -u prepare.py \
  --pretraining_data "$PRETRAINING_DATA" \
  --ontology_path "$ONTOLOGY_PATH" \
  --meds_reader "$MEDS_READER" \
  --time_bins "$TIME_BINS" \
  --value_bins "$VALUE_BINS" \
  --loss_type mtpp_shared \
  --num_final_tasks "$NUM_FINAL_TASKS" \
  --num_threads "$NUM_THREADS" \
  --hf_cache_path "$HF_CACHE_PATH" \
  --model transformer \
  2>&1 | tee "$LOG_DIR/prepare_$(date +%Y%m%d_%H%M%S).log"

# ---------------------------------------------------------------- 2. pretraining
RESUME_ARGS=()
[[ -n "${CHECKPOINT_DIR:-}" ]] && RESUME_ARGS=(--checkpoint_dir "$CHECKPOINT_DIR")

echo "[$(date)] step 2/2: pretraining on GPUs $GPUS ${CHECKPOINT_DIR:+(resuming from $CHECKPOINT_DIR)}"
CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  pretrain.py \
  --pretraining_data "$PRETRAINING_DATA" \
  --meds_reader "$MEDS_READER" \
  --output_dir "$OUTPUT_DIR" \
  --model transformer \
  --loss_type "$LOSS_TYPE" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --hf_run_name "$RUN_NAME" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] done. model -> $OUTPUT_DIR"

# ---------------------------------------------------------------- Pretraining FLOPs / time
# One short run measures both: steps 0-499 under the profiler give FLOPs/token, steps 500-999
# with the profiler off give seconds/token.  Both are extrapolated to one epoch in
# <output_dir>/training_flops_time.json; multiply by the epochs the real run actually reached.
# Keep every other flag identical to the run above, or the numbers won't describe it.
#
# CUDA_VISIBLE_DEVICES is required, not optional: plain `python pretrain.py` on a multi-GPU box
# makes HF Trainer wrap the model in nn.DataParallel across every visible GPU while world_size
# stays 1, so the measured rates would not describe a single GPU.
#
# CUDA_VISIBLE_DEVICES=0 python pretrain.py \
#   --pretraining_data "$PRETRAINING_DATA" \
#   --meds_reader "$MEDS_READER" \
#   --output_dir "$OUTPUT_DIR" \
#   --model transformer \
#   --loss_type mtpp_shared \
#   --hf_run_name mimic_ora_transformer_profile \
#   --profile_steps 1000 \
#   --time_steps 1000 \
#   --max_steps 2000

# ---------------------------------------------------------------- Mamba variant
# The Mamba arm uses a different learning rate, grad-norm clip and depth; it is not a
# like-for-like counterpart of the transformer run above.
#
#   MODEL=mamba, --learning_rate 2e-4, --max_grad_norm 5.0, --n_layers 28
#
# accelerate launch --num_processes "$NUM_PROCESSES" pretrain.py \
#   --pretraining_data "$PRETRAINING_DATA" \
#   --meds_reader "$MEDS_READER" \
#   --output_dir "$PRETRAINING_DATA/mamba" \
#   --model mamba \
#   --loss_type mtpp_shared \
#   --learning_rate 2e-4 \
#   --max_grad_norm 5.0 \
#   --n_layers 28 \
#   --hf_run_name mimic_ora_mamba_2e-4
