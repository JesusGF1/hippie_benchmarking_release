#!/usr/bin/env bash
set -euo pipefail

DATASETS=(
  "a1data_remove_undef"
  "allen_scope_neuropixel_area_subset"
  "cellexplorer_cell_type"
  "dandi_000041_cell_type"
  "dandi_000473_cell_type"
  "dandi_000955_cell_type"
  "hausser_cell_type"
  "hull_cell_type"
  "juxtacellular_mouse_s1_area"
  "lisberger_labeled_cell_type"
)

# --- Hard defaults (can be overridden via env or CLI) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_DATASET_ROOT="${PROJECT_ROOT}/datasets"
DEFAULT_OUT_ROOT="${PROJECT_ROOT}/results/physmap/transductive"

# Env overrides (optional): OUT_ROOT, DATASET_ROOT
OUT_ROOT="${OUT_ROOT:-$DEFAULT_OUT_ROOT}"
DATASET_ROOT="${DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"

# Move to script directory (project root with renv/ and physmap.R)
cd "$(dirname "$0")"

# Sanity checks. renv is optional: if renv/activate.R is present we use it,
# otherwise we fall back to system R packages (caret, dplyr, Seurat, etc.
# must already be installed — see renv.lock for the locked versions).
[[ -f "physmap_script.r" ]] || { echo "ERROR: physmap_script.r not found."; exit 1; }
if [[ -f "renv/activate.R" ]]; then
  R_INVOCATION='source("renv/activate.R"); source("physmap_script.r")'
else
  echo "INFO: renv/activate.R not found — using system R packages."
  R_INVOCATION='source("physmap_script.r")'
fi

# Loop through datasets
for dataset in "${DATASETS[@]}"; do
  echo "=== Running experiment for dataset: $dataset ==="

  OUT_DIR="${OUT_ROOT}/${dataset}/physmap"

  # Create output directory if it doesn't exist
  mkdir -p "${OUT_DIR}"

  # Build argument list for this dataset
  args=()
  if [[ $# -gt 0 ]]; then
    args=( "$@" )
  fi
  has_key() {
    local key="$1"
    if [[ ${#args[@]} -gt 0 ]]; then
      for a in "${args[@]}"; do
        [[ "$a" == "$key="* ]] && return 0
      done
    fi
    return 1
  }

  if ! has_key "--out-dir"; then
    args+=( "--out-dir=${OUT_DIR}" )
  fi
  if ! has_key "--dataset-root"; then
    args+=( "--dataset-root=${DATASET_ROOT}" )
  fi
  if ! has_key "--dataset-folder"; then
    args+=( "--dataset-folder=${dataset}" )
  fi

  # Run under renv if available, otherwise system R
  R -q -e "${R_INVOCATION}" --args "${args[@]}"

  echo "=== Finished dataset: $dataset ==="
done
