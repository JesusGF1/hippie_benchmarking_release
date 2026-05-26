#!/usr/bin/env bash
# NEMO Benchmark Evaluation Script
# Runs NEMO evaluation on C4 database H5 files (requires spike times for 3D ACG)
set -euo pipefail

# C4 Database H5 files - loaded directly from S3 or local cache
# These are the original H5 files from https://www.c4-database.com/apps/download
# IMPORTANT: NEMO requires H5 format with 3D ACGs, NOT HIPPIE CSV format

# Optional S3 location for C4 database files. When unset, the script expects
# the H5 files to be available locally under ${C4_LOCAL_DIR} (default
# ./datasets/c4_h5/) — set the env vars below only if you fetch from S3.
S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-}"
S3_ENDPOINT="${S3_ENDPOINT:-}"
C4_LOCAL_DIR="${C4_LOCAL_DIR:-./datasets/c4_h5}"

# Dataset configurations: dataset_name -> H5 filename
declare -A DATASETS=(
    ["hull"]="C4_database_hull_labelled.h5"
    ["hausser"]="C4_database_hausser_labelled.h5"
    ["lisberger"]="C4_database_lisberger_labelled.h5"
    ["medina"]="C4_database_medina_labelled.h5"
)

# --- Hard defaults (can be overridden via env or CLI) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DEFAULT_OUT_ROOT="${PROJECT_ROOT}/results/benchmark_cv"

# Env overrides (optional)
OUT_ROOT="${OUT_ROOT:-$DEFAULT_OUT_ROOT}"

# NEMO hyperparameters
Z_DIM="${Z_DIM:-512}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
N_FOLDS="${N_FOLDS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
TEMPERATURE="${TEMPERATURE:-0.5}"

# Move to script directory
cd "$SCRIPT_DIR"

# Check that the evaluation script exists
[[ -f "nemo_benchmark_evaluation.py" ]] || { echo "ERROR: nemo_benchmark_evaluation.py not found."; exit 1; }

echo "=============================================="
echo "NEMO Benchmark Evaluation"
echo "=============================================="
if [[ -n "${S3_BUCKET}" ]]; then
    echo "S3 bucket: ${S3_BUCKET}/${S3_PREFIX}"
else
    echo "Local H5 dir: ${C4_LOCAL_DIR}"
fi
echo "Output root: ${OUT_ROOT}"
echo "Z dim: ${Z_DIM}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "N folds: ${N_FOLDS}"
echo "=============================================="

# Function to run NEMO evaluation for a single dataset
run_nemo_evaluation() {
    local dataset_name="$1"
    local h5_file="${DATASETS[$dataset_name]}"
    local out_dir="${OUT_ROOT}/${dataset_name}_cell_type/nemo"

    # Resolve dataset path: when an S3 bucket is configured, pass --s3-path.
    # Otherwise the python script reads the H5 from NEMO_DATASETS_DIR (default
    # ./datasets in comparison_methods/nemo/src/celltype_ibl/params/config.py).
    local extra_args=""
    if [[ -n "${S3_BUCKET}" ]]; then
        extra_args="--s3-path s3://${S3_BUCKET}/${S3_PREFIX}/${h5_file}"
        echo "Source: S3 (s3://${S3_BUCKET}/${S3_PREFIX}/${h5_file})"
    else
        export NEMO_DATASETS_DIR="${C4_LOCAL_DIR}"
        echo "Source: local (${C4_LOCAL_DIR}/${h5_file})"
    fi

    echo ""
    echo "=== Processing: $dataset_name ==="
    echo "H5 file: $h5_file"

    # Create output directory
    mkdir -p "${out_dir}"

    # Run NEMO evaluation for each fold
    for fold in $(seq 0 $((N_FOLDS - 1))); do
        echo ""
        echo "--- Running fold ${fold}/${N_FOLDS} for ${dataset_name} ---"

        python nemo_benchmark_evaluation.py \
            --dataset-file "${h5_file}" \
            ${extra_args} \
            --out-dir "${out_dir}" \
            --cv-fold "${fold}" \
            --n-cv-folds "${N_FOLDS}" \
            --z-dim "${Z_DIM}" \
            --epochs "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --learning-rate "${LEARNING_RATE}" \
            --temperature "${TEMPERATURE}" \
            || { echo "WARNING: NEMO evaluation failed for ${dataset_name} fold ${fold}"; continue; }
    done

    # Aggregate results across folds
    echo ""
    echo "Aggregating results for ${dataset_name}..."
    python -c "
import pandas as pd
import os

out_dir = '${out_dir}'
n_folds = ${N_FOLDS}

# Collect all fold predictions
all_predictions = []
for fold in range(n_folds):
    pred_file = os.path.join(out_dir, f'fold_{fold}', 'transductive_predictions.csv')
    if os.path.exists(pred_file):
        df = pd.read_csv(pred_file)
        df['fold'] = fold
        all_predictions.append(df)

if all_predictions:
    combined = pd.concat(all_predictions, ignore_index=True)
    combined.to_csv(os.path.join(out_dir, 'nemo_transductive_predictions.csv'), index=False)
    print(f'Combined {len(all_predictions)} folds into nemo_transductive_predictions.csv')

    # Compute overall metrics
    from sklearn.metrics import balanced_accuracy_score, accuracy_score
    acc = accuracy_score(combined['true'], combined['pred'])
    bal_acc = balanced_accuracy_score(combined['true'], combined['pred'])
    print(f'Overall Accuracy: {acc:.4f}')
    print(f'Overall Balanced Accuracy: {bal_acc:.4f}')
else:
    print('WARNING: No prediction files found to aggregate')
"

    echo "=== Finished: ${dataset_name} ==="
    return 0
}

# Parse command line arguments
SELECTED_DATASETS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            SELECTED_DATASETS+=("$2")
            shift 2
            ;;
        --all)
            SELECTED_DATASETS=("${!DATASETS[@]}")
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset NAME] [--all]"
            echo "Available datasets: ${!DATASETS[*]}"
            exit 1
            ;;
    esac
done

# Default to all datasets if none specified
if [[ ${#SELECTED_DATASETS[@]} -eq 0 ]]; then
    SELECTED_DATASETS=("${!DATASETS[@]}")
fi

# Loop through selected datasets
for dataset in "${SELECTED_DATASETS[@]}"; do
    if [[ -v "DATASETS[$dataset]" ]]; then
        run_nemo_evaluation "$dataset" || true
    else
        echo "WARNING: Unknown dataset '$dataset', skipping..."
    fi
done

echo ""
echo "=============================================="
echo "NEMO Benchmark Evaluation Complete"
echo "=============================================="
echo "Results saved to: ${OUT_ROOT}"
echo ""
