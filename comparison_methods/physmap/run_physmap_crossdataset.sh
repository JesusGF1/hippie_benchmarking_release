#!/usr/bin/env bash
set -euo pipefail

# ----------------------- Defaults (overridable) -----------------------
: "${TRAIN_DATASET:=hull_cell_type}"
: "${PREDICT_DATASET:=lisberger_labeled_cell_type}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${DEFAULT_DATASET_ROOT:=${PROJECT_ROOT}/datasets}"
: "${DEFAULT_OUT_ROOT:=${PROJECT_ROOT}/results/physmap/cross_dataset}"

# Env overrides (optional): OUT_ROOT, DATASET_ROOT
OUT_ROOT="${OUT_ROOT:-$DEFAULT_OUT_ROOT}"
DATASET_ROOT="${DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"

# ----------------------- Locate project + script -----------------------
cd "$(dirname "$0")"

# Prefer .R; fall back to .r for convenience
RSCRIPT_FILE="physmap_knn_crossdataset.R"
if [[ ! -f "$RSCRIPT_FILE" ]]; then
  if [[ -f "physmap_knn_crossdataset.r" ]]; then
    RSCRIPT_FILE="physmap_knn_crossdataset.r"
  else
    echo "ERROR: physmap_knn_crossdataset.R (or .r) not found." >&2
    exit 1
  fi
fi

[[ -f "renv/activate.R" ]] || { echo "ERROR: renv/activate.R not found."; exit 1; }

# ----------------------- Arg handling helpers --------------------------
args=( "$@" )

has_key() {
  local key="$1"
  for a in "${args[@]:-}"; do
    [[ "$a" == "$key="* ]] && return 0
  done
  return 1
}

append_if_missing() {
  local key="$1" value="$2"
  if ! has_key "$key"; then
    args+=( "${key}=${value}" )
  fi
}

# ----------------------- Compose run-specific OUT ----------------------
PAIR_TAG="${TRAIN_DATASET}__to__${PREDICT_DATASET}"
OUT_DIR="${OUT_ROOT}/${PAIR_TAG}/physmap"

# ----------------------- Fill required --args --------------------------
append_if_missing "--out-dir"        "$OUT_DIR"
append_if_missing "--dataset-root"   "$DATASET_ROOT"
append_if_missing "--train-folder"   "$TRAIN_DATASET"
append_if_missing "--predict-folder" "$PREDICT_DATASET"

echo "=== Running physmap KNN cross-dataset ==="
echo "  Script        : ${RSCRIPT_FILE}"
echo "  Train folder  : ${TRAIN_DATASET}"
echo "  Predict folder: ${PREDICT_DATASET}"
echo "  Dataset root  : ${DATASET_ROOT}"
echo "  Out dir       : ${OUT_DIR}"
echo "  Extra args    : ${args[*]:-<none>}"
echo "========================================="

# Ensure output directory exists early (nice for logs)
mkdir -p "$OUT_DIR"

# ----------------------- Execute under renv ----------------------------
# We use base R so we can pass --args to the sourced script.
START_TIME=$(date +%s)
R -q -e 'source("renv/activate.R"); source("'"$RSCRIPT_FILE"'")' --args "${args[@]}"
END_TIME=$(date +%s)

ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=== Done: ${PAIR_TAG} ==="
echo "Total execution time: ${MINUTES}m ${SECONDS}s (${ELAPSED} seconds)"
echo ""
