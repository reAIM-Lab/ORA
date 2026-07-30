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

```bash
pip install -e .
pip install xformers
```

Make sure to also install the correct gpu enabled version of PyTorch.

The tabular baselines additionally need `xgboost` and `scikit-survival`:

```bash
pip install xgboost scikit-survival
```


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
| `mtpp_shared` | Shared MTPP - MTPP with shared parameters across tasks. **This is ORA.** | The proposed objective |
| `ntp` | NTP (Next Token Prediction) - CLMBR-style next code prediction | Language model-style pretraining |
| `survivehr` | SurvivEHR - DeSurv competing-risk parameterization with per-code Gaussian value heads | Foundation-model baseline |
| `icarefm` | ICareFM - threshold-conditioned hourly hazard bins with masked BCE on first threshold crossing | Foundation-model baseline |
| `mtpp_factorized` | Ablation of ORA that predicts time and value **separately** rather than jointly | Ablation |
| `mtpp_value` | Ablation variant with a dedicated value head | Ablation |

**Note:** `mtpp` and `mtpp_shared` support the `--value_bins` parameter for discretizing continuous lab values.

### Baseline objectives

`survivehr` and `icarefm` are our own reimplementations: neither method publishes weights or a usable
implementation for MEDS-formatted data. They are implemented against the same vocabulary, tokenizer and
backbone as every other objective here, so that the pretraining loss is the only thing that varies.

- **SurvivEHR** is ported against the officially released code: the identical DeSurv competing-risk
  parameterization and per-code Gaussian value heads. See `src/ora/models/tasks/survivehr_task.py` and
  `src/ora/models/architecture/survivehr_head.py`.
- **ICareFM** has no public implementation, so it follows the paper's supplementary description:
  48 hourly hazard bins, prediction heads conditioned on the queried threshold and its direction,
  thresholds resampled at every training step, and a masked binary cross-entropy loss on the first
  threshold crossing. See `src/ora/models/tasks/icarefm_task.py` and `ICareFMTaskHead` in
  `src/ora/models/architecture/embedding.py`.

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
| `--loss_type` | Yes | Type of pretraining task: `motor`, `tpp`, `mtpp`, `mtpp_shared`, `mtpp_value`, `ntp`, `survivehr`, or `icarefm` |
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
| `--loss_type` | Yes | Type of pretraining loss: `motor`, `tpp`, `mtpp_shared`, `mtpp_factorized`, `mtpp_value`, `ntp`, `survivehr`, or `icarefm` (must match prepare.py) |
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

**Script:** `evaluation/classification_tte/run.sh`

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

**Script:** `evaluation/classification_tte/run.sh`

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
| `--truncation` | Yes | 330 for mimic and 1825 for institutional |
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
| `--tasks` | Yes | Comma-separated list of regression targets (MIMIC: `"pao2,platelets,glucose,creatinine"`; institutional: `"spo2,platelets,glucose,creatinine"`) |
| `--regression` | Yes | Flag to enable regression mode (no value needed) |
| `--device` | Yes | GPU device (e.g., `cuda:0`) |
| `--cache_path` | Yes | HuggingFace datasets cache directory |
| `--tokens_per_batch` | Yes | Maximum tokens per batch (default: 65536) |
| `--min_subjects_per_batch` | Yes | Minimum patients per batch = tokens_per_batch/8192 (default: 8) |
| `--num_proc` | No | Number of parallel processes (default: 24) |
| `--use_ntp` | Yes | Whether to use NTP mode: `yes` or `no` |

## 5.4. Evaluation Layout

`src/ora/omop_meds_tutorial/evaluation/` is organised by task family. Shared machinery lives in
`common/`; each task family owns its probe, its analysis scripts and its significance test.

Each task family holds its driver next to the probe it drives.

```
evaluation/
  common/             generate_labels, generate_features, eval_utils, survival_utils
  classification_tte/ run.sh + linear_probe.py     ORA probe: classification and TTE
  regression/         run_regression.sh + probe.py
  baseline/           the tabular baselines (XGBoost / DeepHit / most-recent-value)
  p_test/             classification/, tte/, regression/   paired bootstrap tests
  few_shot/           few-shot linear probing
  run_icare.sh        convenience wrapper: one model across all task families
```

## 5.5. Tabular Baselines

`evaluation/baseline/` covers all three task families. `run_all.py` is the only entry point:

| `--task` | Model | Module |
|---|---|---|
| `classification` | XGBoost | `classification.py` |
| `tte` | DeepHit | `tte.py` |
| `regression` | most-recent value | `regression.py` |

Features come from `feature_extraction/`, which chooses between two feature sets with
`--value_stats`:

- **`yes`** (default) — code counts **plus** the MEDS-Tab numeric aggregation set (`code/count`,
  `value/count`, `value/sum`, `value/sum_sqd`, `value/min`, `value/max` over `[7d, full]` windows).
- **`no`** — code counts only. `CountFeaturizer` under default flags drops every numeric-valued event
  from the matrix entirely, so this is the count-only baseline in which the model never learns that a
  lab was ordered.

The two write to separate roots, so both rows can be reported side by side.

```bash
BASE=ora.omop_meds_tutorial.evaluation.baseline
python -m $BASE.feature_extraction.validate_featurizer --num_subjects 200
bash src/ora/omop_meds_tutorial/evaluation/baseline/feature_extraction/extract.sh
python -m $BASE.run_all --task all --n_jobs 24 --truncation 300
```

`--truncation` must match the horizon the foundation-model probe is run with
(`classification_tte/run.sh --truncation`), or the C-indices are not comparable. Both default to 300
(`baseline/common.py:DEFAULT_TRUNCATION`). See `baseline/README.md` for the feature specification and
hyperparameter grids.

## 5.6. Significance Testing

Paired bootstrap confidence intervals for the difference between ORA and each baseline
(10,000 resamples, seed 42), using shared bootstrap indices so the pairing is valid:

| Script | Produces |
|---|---|
| `p_test/classification/classification_test.py` | ΔAUROC 95% CIs |
| `p_test/tte/tte_test.py`, `p_test/tte/compare.py` | ΔC-index 95% CIs |
| `p_test/regression/regression_test.py` | ΔR² 95% CIs |

## 5.7. Compute Accounting

Pretraining and feature extraction record FLOPs and wall time:

- `pretrain.py --profile_steps N --time_steps M` runs the torch profiler over the first `N` training
  steps to measure FLOPs/token, then times the next `M` steps with the profiler off to measure
  seconds/token. Results are written next to the checkpoints.
- `evaluation/baseline/collect_compute.py` merges the per-task extraction and training JSON
  (wall-clock, peak RSS, column budget, density) into a single table.

All models in the paper were pretrained on 4× NVIDIA H100 NVL (96 GB VRAM).

## Evaluation Outputs

All evaluation scripts generate results in their respective `--output_dir`:

- **Classification tasks**: AUROC, AUPRC
- **Time-to-Event tasks**: C-index, Int-Brier Score
- **Regression tasks**: RMSE, MAE, R² scores

Results are saved as CSV files and can be further analyzed for model performance.
