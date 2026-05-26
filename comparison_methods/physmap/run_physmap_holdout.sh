#!/usr/bin/env bash
set -euo pipefail

# Default dataset for holdout (allen_scope_neuropixel_area only)
DEFAULT_DATASET="allen_scope_neuropixel_area_subset"

# --- Hard defaults (can be overridden via env or CLI) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_DATASET_ROOT="${PROJECT_ROOT}/datasets"
DEFAULT_OUT_ROOT="${PROJECT_ROOT}/results/physmap/holdout"

# Holdout parameters
DEFAULT_N_FOLDS=5
DEFAULT_MIN_CELLS=50

# Env overrides (optional): OUT_ROOT, DATASET_ROOT
OUT_ROOT="${OUT_ROOT:-$DEFAULT_OUT_ROOT}"
DATASET_ROOT="${DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"

# Move to script directory (project root with renv/ and physmap_script_holdout.r)
cd "$(dirname "$0")"

# Sanity checks
[[ -f "renv/activate.R" ]] || { echo "ERROR: renv/activate.R not found."; exit 1; }
[[ -f "physmap_script_holdout.r" ]] || { echo "ERROR: physmap_script_holdout.r not found."; exit 1; }

# Parse arguments to get fold parameters
n_folds=$DEFAULT_N_FOLDS
specific_fold=""
min_cells=$DEFAULT_MIN_CELLS
dataset=$DEFAULT_DATASET

for arg in "$@"; do
  case $arg in
    --n-holdout-folds=*)
      n_folds="${arg#*=}"
      ;;
    --holdout-fold=*)
      specific_fold="${arg#*=}"
      ;;
    --min-cells-per-animal=*)
      min_cells="${arg#*=}"
      ;;
    --dataset-folder=*)
      dataset="${arg#*=}"
      ;;
  esac
done

echo "=== Running PHYSMAP holdout experiments ==="
echo "Dataset: $dataset"
echo "Number of folds: $n_folds"
echo "Minimum cells per animal: $min_cells"
echo "Dataset root: $DATASET_ROOT"
echo "Output root: $OUT_ROOT"
echo

# Helper function to check if argument exists
has_key() {
  local key="$1"
  for a in "$@"; do
    [[ "$a" == "$key="* ]] && return 0
  done
  return 1
}

# Function to run a single fold
run_fold() {
  local fold=$1
  echo "=== Running holdout fold $fold/$((n_folds-1)) for dataset: $dataset ==="

  # Build argument list for this fold
  args=( "$@" )  # Start with all passed arguments

  # Override or add specific parameters
  local updated_args=()
  local fold_set=false
  local dataset_set=false
  local out_dir_set=false
  local dataset_root_set=false

  for arg in "${args[@]}"; do
    case $arg in
      --holdout-fold=*)
        updated_args+=( "--holdout-fold=$fold" )
        fold_set=true
        ;;
      --dataset-folder=*)
        updated_args+=( "--dataset-folder=$dataset" )
        dataset_set=true
        ;;
      --out-dir=*)
        updated_args+=( "--out-dir=$OUT_ROOT" )
        out_dir_set=true
        ;;
      --dataset-root=*)
        updated_args+=( "--dataset-root=$DATASET_ROOT" )
        dataset_root_set=true
        ;;
      *)
        updated_args+=( "$arg" )
        ;;
    esac
  done

  # Add missing parameters
  if ! $fold_set; then
    updated_args+=( "--holdout-fold=$fold" )
  fi
  if ! $dataset_set; then
    updated_args+=( "--dataset-folder=$dataset" )
  fi
  if ! $out_dir_set; then
    updated_args+=( "--out-dir=$OUT_ROOT" )
  fi
  if ! $dataset_root_set; then
    updated_args+=( "--dataset-root=$DATASET_ROOT" )
  fi

  echo "Running: R -q -e 'source(\"renv/activate.R\"); source(\"physmap_script_holdout.r\")' --args ${updated_args[*]}"

  # Run under renv
  R -q -e 'source("renv/activate.R"); source("physmap_script_holdout.r")' --args "${updated_args[@]}"

  echo "=== Finished holdout fold $fold for dataset: $dataset ==="
  echo
}

# Run specific fold or all folds
if [[ -n "$specific_fold" ]]; then
  # Run specific fold
  run_fold "$specific_fold" "$@"
else
  # Run all folds
  for fold in $(seq 0 $((n_folds-1))); do
    run_fold "$fold" "$@"
  done
fi

echo "=== All PHYSMAP holdout experiments completed ==="
echo "Results saved to: $OUT_ROOT/$dataset/"