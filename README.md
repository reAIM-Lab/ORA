# ORA
Implementation for One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models


Credit: this project is adapted from an upstream open-source EHR framework: https://github.com/som-shahlab/femr

Step 1. Converting into meds_reader
------------------------

The ORA library uses the [meds_reader](https://github.com/EthanSteinberg/meds_reader) utility for processing MEDS data. This requires a second preprocessing step

```bash
pip install meds_reader==0.1.3
```

```bash
export OMOP_MEDS = ""
export OMOP_MEDS_READER = ""
export PRETRAINING_DATA = ""
export ATHENA_DATA = ""

```

```bash
meds_reader_convert $OMOP_MEDS $OMOP_MEDS_READER --num_threads 16
```

Step 2. Downloading Athena
-------------------------

ORA uses OHDSI's Athena tool for ontology processing. Go to https://athena.ohdsi.org/ and download the folder.

You can create an account for free.

Note: Make sure to run the CPT4 fixer script in the Athena download before continuing!

Step 3. Installing ORA
------------------------

Finally, you need to install ORA.

This code currently uses a branch of ORA, mimic_tutorial, in order to function. However, soon these changes will be merged upstream into ora 2.4

```bash
git clone https://github.com/reAIM-Lab/EHR_TTE.git
cd EHR_TTE
pip install -e .
pip install xformers
```

Make sure to also install the correct gpu enabled version of PyTorch


Step 4. Preparing For Pretraining
------------------------

## Overview

The pretraining pipeline consists of two main scripts:
1. **prepare.py** - Prepares data, trains tokenizer, creates tasks, and generates training batches
2. **pretrain.py** - Trains the foundation model using the prepared data

## Loss Types Explained

The framework supports multiple pretraining objectives:

| Loss Type | Description | Use Case |
|-----------|-------------|----------|
| `motor` | MOTOR (Multi-task Temporal Risk) - Predicts time until multiple medical events | General purpose foundation model |
| `tpp` | TPP (Temporal Point Process) - Models timing of medical events | Time-focused modeling |
| `mtpp` | MTPP (Multivariate TPP) - Models both timing and lab values | When lab values are important |
| `mtpp_shared` | Shared MTPP - MTPP with shared parameters across tasks | Memory-efficient MTPP variant |
| `ntp` | NTP (Next Token Prediction) - CLMBR-style next code prediction | Language model-style pretraining |

**Note:** `mtpp` and `mtpp_shared` support the `--value_bins` parameter for discretizing continuous lab values.

## Quick Start Workflow

```bash
# Step 1: Prepare data, tokenizer, and tasks
python prepare.py \
  --pretraining_data /path/to/output \
  --ontology_path /path/to/ontology \
  --time_bins discretized_time_bins \
  --value_bins discretized_value_bins \
  --loss_type loss_type \
  --num_final_tasks pretraining_task_num \
  --hf_cache_path cache_path \ 
  --num_threads num_threads \
  --meds_reader /path/to/meds_reader

# Step 2: Train the model
accelerate launch \
  --num_processes 4 \
  pretrain.py \
  --pretraining_data /path/to/output \
  --meds_reader /path/to/meds_reader \
  --output_dir /path/to/output/transformer \
  --model transformer \
  --hf_run_name my_experiment \
  --loss_type mtpp_shared
```

### 4.1. Running prepare.py

The `prepare.py` script performs the following:
- Creates train/validation/test splits
- Generates or loads an ontology from Athena
- Trains a hierarchical tokenizer
- Creates pretraining tasks (MOTOR, TPP, MTPP, or NTP)
- Converts patient data into batches for training

**Example Usage:**

```bash
python prepare.py \
  --pretraining_data $Path_to_pretraining_data \
  --ontology_path ontology_path \
  --time_bins 8 \
  --value_bins 5 \
  --loss_type mtpp \
  --num_final_tasks 6100 \
  --num_threads 32 \
  --hf_cache_path $Path_to_hf_cache \
  --meds_reader $Path_to_meds_reader
```

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--pretraining_data` | Yes | **Output directory** that will store all generated artifacts: train/val splits, tokenizer, task definitions, and training/evaluation batches |
| `--ontology_path` | Yes | Directory path where the ontology file (ontology.pkl) will be saved or loaded from |
| `--meds_reader` | Yes | Path to the MEDS reader database containing the input patient data |
| `--athena_path` | No | Path to Athena ontology data for creating the medical code ontology (optional if ontology already exists) |
| `--time_bins` | No | Number of time bins for temporal discretization (default: 8) |
| `--value_bins` | No | Number of bins for discretizing lab values in MTPP tasks (default: 10) |
| `--loss_type` | Yes | Type of pretraining task: `motor`, `tpp`, `mtpp`, or `ntp` |
| `--num_final_tasks` | No | Number of final prediction tasks for MTPP (default: 8192) |
| `--num_threads` | No | Number of parallel threads for data processing (default: 16) |
| `--hf_cache_path` | No | HuggingFace datasets cache directory |
| `--vocab_size` | No | Tokenizer vocabulary size (default: 16384) |
| `--tokens_per_batch` | No | Maximum tokens per training batch (default: 8192) |
| `--value_count_threshold` | No | Minimum occurrence count for lab values to be included (default: 1000) |
| `--motor_codes_to_skip` | No | Path to parquet file containing medical codes to exclude from pretraining |

**Output Structure:**

After running prepare.py, the `--pretraining_data` directory will contain:
```
pretraining_data/
├── main_split.csv              # Train/test patient splits
├── ontology.pkl                # Medical ontology
├── tokenizer/                  # Trained tokenizer files
├── {loss_type}_task.pkl        # Pretraining task definition
├── train_batches/              # Training data batches (transformer/local attention)
├── val_batches/                # Validation data batches (transformer/local attention)
├── train_batches_mamba/        # Training batches optimized for Mamba (≤8192 per sample, ≤16384 per batch)
└── val_batches_mamba/          # Validation batches optimized for Mamba
```

### 4.2. Running pretrain.py

The `pretrain.py` script trains a foundation model (Transformer or Mamba) using the prepared data. It supports multi-GPU training via Hugging Face Accelerate.

Note: When running prepare.py, pass `--model transformer` to build transformer-style batches or `--model mamba` to build Mamba-optimized batches only. Pretrain automatically loads the correct directory based on `--model`.

**Example Usage (Multi-GPU):**

```bash
# Set visible GPUs (optional)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Launch training with accelerate
accelerate launch \
  --num_processes 4 \
  pretrain.py \
  --pretraining_data $Path_to_pretraining_data \
  --meds_reader $Path_to_meds_reader \
  --output_dir $Path_to_output_dir \
  --model transformer \
  --hf_run_name mimic_mtpp_shared_aligned_task_bin_5 \
  --loss_type mtpp_shared
```

**Example Usage (Single GPU):**

```bash
python pretrain.py \
  --pretraining_data $Path_to_pretraining_data \
  --meds_reader $Path_to_meds_reader \
  --output_dir $Path_to_output_dir \
  --model transformer \
  --hf_run_name my_run \
  --loss_type mtpp \
  --per_device_train_batch_size 1
```

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--pretraining_data` | Yes | **Same directory from prepare.py** containing tokenizer, task, and batches |
| `--meds_reader` | Yes | Path to MEDS reader database (same as prepare.py) |
| `--output_dir` | Yes | Directory where model checkpoints will be saved during training |
| `--model` | Yes | Model architecture: `transformer` or `mamba` |
| `--loss_type` | Yes | Type of pretraining loss: `motor`, `tpp`, `mtpp`, `mtpp_shared`, or `ntp` (must match prepare.py) |
| `--hf_run_name` | Yes | Name for the training run (displayed in WandB) |
| `--checkpoint_dir` | No | Path to checkpoint to resume training from |
| `--learning_rate` | No | Learning rate for optimizer (default: 1e-5) |
| `--n_layers` | No | Number of transformer/mamba layers (default: 11) |
| `--n_epochs` | No | Number of training epochs (default: 50) |
| `--per_device_train_batch_size` | No | Training batch size per GPU (default: 1) |
| `--per_device_eval_batch_size` | No | Evaluation batch size per GPU (default: 1) |
| `--linear_interpolation` | No | Use linear interpolation for time modeling (default: False) |

**Important Notes:**

1. **Loss Type Matching**: The `--loss_type` in pretrain.py must match the task created in prepare.py
   - If you ran prepare.py with `--loss_type mtpp`, you must use `--loss_type mtpp` or `--loss_type mtpp_shared` in pretrain.py

2. **Multi-GPU Training**: Use `accelerate launch` for distributed training:
   ```bash
   accelerate launch --num_processes N pretrain.py [args]
   ```

3. **Checkpoints**: The model automatically saves checkpoints to `--output_dir` after each epoch. The best checkpoint (lowest validation loss) is saved separately.

4. **Monitoring**: Training metrics are logged to Weights & Biases (WandB) using the `--hf_run_name` identifier.

5. **Memory Requirements**: Foundation model pretraining typically requires high-memory GPUs (24GB+ recommended). Adjust `--per_device_train_batch_size` based on your GPU memory.

**Output:**

Training produces checkpoints in the `--output_dir`:
```
output_dir/
├── checkpoint-1000/
├── checkpoint-2000/
├── ...
└── checkpoint-best/          # Best performing checkpoint
```

Step 5. Model Evaluation
------------------------

After pretraining your model, you can evaluate it on downstream tasks using the evaluation scripts. The framework supports three types of evaluation:

## 5.1. Outcome Classification

Evaluates binary classification tasks (e.g., predicting patient outcomes).

**Script:** `evaluation/run.sh`

**Example Usage:**

```bash
# Mimic
bash run.sh \
  --meds_reader        $Path_to_meds_reader  \
  --num_proc           32 \
  --model_path         $Path_to_model_checkpoint \
  --model_name         mamba_mtpp \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --use_ntp            no \
  --cohort_type        outcome \
  --cache_path           $Path_to_hf_cache \
  --min_subjects_per_batch 8 \
  --ontology_path       $Path_to_ontology \
  --main_split_path     $Path_to_main_split_csv \
  --output_dir   $Path_to_output_dir  \
  --classification_cohort $Path_to_classification_cohort


```

## 5.2. Phenotype Classification & Time-to-Event Prediction

Evaluates phenotype classification and survival analysis (time-to-event) tasks.

**Script:** `evaluation/run.sh`

**Example Usage**

```bash
# export CUDA_VISIBLE_DEVICES=2
# export CUDA_VISIBLE_DEVICES=1
# mimic
bash run.sh \
  --meds_reader        $Path_to_meds_reader \
  --num_proc           24 \
  --model_path         $Path_to_model_checkpoint \
  --model_name         mamba_ntp_2e-4 \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --use_ntp            yes \
  --cohort_type        phenotype \
  --cache_path        $Path_to_hf_cache \
  --min_subjects_per_batch 8 \
  --truncation  330 \
  --observation_window 730 \
  --ontology_path       $Path_to_ontology \
  --main_split_path     $Path_to_main_split_csv \
  --output_dir   $Path_to_output_dir  \
  --classification_cohort  $Path_to_classification_cohort \
  --tte_cohort $Path_to_tte_cohort

```
### Parameters for run.sh

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--meds_reader` | Yes | Path to MEDS reader database (same as pretraining) |
| `--model_path` | Yes | Path to trained model checkpoint (e.g., `best_159744` or `checkpoint-1000`) |
| `--model_name` | Yes | Model type: `transformer_mtpp`, `transformer_motor`, `transformer_ntp`, or `mamba_*` variants |
| `--output_dir` | Yes | Directory where evaluation results will be saved |
| `--ontology_path` | Yes | Path to ontology.pkl file (from prepare.py) |
| `--main_split_path` | Yes | Path to main_split.csv file (from prepare.py) |
| `--cohort_type` | Yes | Type of task: `outcome`, `phenotype`, or other cohort types |
| `--classification_cohort` | Yes | Path to classification task cohort directory (required for classification tasks) |
| `--tte_cohort` | Yes | Path to time-to-event cohort directory (required for TTE tasks) |
| `--observational window` | No| Observational Window before the prediction time. Use 730 for phenotype prediction |
| `--truncation` | Yes | 330 for mimic and 1825 for cumc |
| `--cache_path` | Yes | Path to store hugginface datasets. You can set a dir like $Path_to_hf_cache |
| `--device` | Yes | GPU device to use (e.g., `cuda:0`, `cuda:1`) |
| `--tokens_per_batch` | No | Maximum tokens per batch (default: 65536). Adjust based on GPU memory |
| `--min_subjects_per_batch` | No | Minimum patients per batch (default: 8) |
| `--num_proc` | No | Number of parallel processes for data loading (default: 24) |
| `--use_ntp` | No | Whether to use NTP mode: `yes` or `no` (default: `no`) |

## 5.3. Regression Tasks

Evaluates continuous value prediction tasks (e.g., predicting lab values).

**Script:** `evaluation/regression/run_regression.sh`

**Example Usage:**
<!-- export CUDA_VISIBLE_DEVICES=1 -->
```bash
bash run_regression.sh \
  --meds_reader        $Path_to_meds_reader \
  --num_proc           24 \
  --model_path         $Path_to_model_checkpoint \
  --model_name         transformer-unmark \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --min_subjects_per_batch 8 \
  --ontology_path      $Path_to_ontology \
  --main_split_path    $Path_to_main_split_csv \
  --cohort_dir   $Path_to_regression_cohort \
  --output_dir   $Path_to_output_dir \
  --use_ntp no \
  --cache_path $Path_to_hf_cache \
  --tasks "bilirubin,creatinine" \
  --regression 

  # pao2,platelets

```
### Parameters for run_regression.sh

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--meds_reader` | Yes | Path to MEDS reader database |
| `--model_path` | Yes | Path to trained model checkpoint |
| `--model_name` | Yes | Model type (e.g., `transformer_mtpp`) |
| `--cohort_dir` | Yes | Directory containing regression task cohort data |
| `--output_dir` | Yes | Directory where regression results will be saved |
| `--ontology_path` | Yes | Path to ontology.pkl file |
| `--main_split_path` | Yes | Path to main_split.csv file |
| `--tasks` | Yes | Comma-separated list of regression targets (MIMIC: `"pao2,platelets,bilirubin,creatinine"` CUMC: `"spo2,platelets,bilirubin,creatinine"`) |
| `--regression` | Yes | Flag to enable regression mode (no value needed) |
| `--device` | Yes | GPU device (e.g., `cuda:0`) |
| `--cache_path` | Yes | HuggingFace datasets cache directory |
| `--tokens_per_batch` | Yes | Maximum tokens per batch (default: 65536) |
| `--min_subjects_per_batch` | Yes | Minimum patients per batch = tokens_per_batch/8192 (default: 8) |
| `--num_proc` | No | Number of parallel processes (default: 24) |
| `--use_ntp` | Yes | Whether to use NTP mode: `yes` or `no` |

## Evaluation Outputs

All evaluation scripts generate results in their respective `--output_dir`:

- **Classification tasks**: AUROC, AUPRC
- **Time-to-Event tasks**: C-index, Int-Brier Score
- **Regression tasks**: RMSE, MAE, R² scores

Results are saved as CSV files and can be further analyzed for model performance.
