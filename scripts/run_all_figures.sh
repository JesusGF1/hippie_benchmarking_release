#!/usr/bin/env bash
#
# Regenerate every paper figure from the cached predictions in results/.
# No GPU, no S3, no cluster needed; everything reads from the local repo.
#
# Usage:
#   conda activate hippie
#   bash scripts/run_all_figures.sh
#
# Outputs land in figures/<script-name>/ next to the pre-generated reference
# images bundled with the repo; compare new outputs against them to confirm
# consistency.

set -u
cd "$(dirname "$0")/.."

LOG_DIR="logs/figure_run_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"
REPORT="$LOG_DIR/report.tsv"
printf "script\texit_code\tduration_s\n" > "$REPORT"

# Run a script; record its exit code and runtime.
run() {
  local script="$1"
  shift
  local name
  name="$(basename "$script" .py)"
  local log="$LOG_DIR/${name}.log"
  echo "=== [$name] ===" | tee -a "$log"
  local t0
  t0=$(date +%s)
  python "$script" "$@" >>"$log" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  printf "%s\t%d\t%d\n" "$name" "$rc" "$dt" >> "$REPORT"
  if [ "$rc" -ne 0 ]; then
    echo "[$name] FAILED (exit $rc) — see $log"
  else
    echo "[$name] ok (${dt}s)"
  fi
}

# Figure 2 — ablation + hyperparameter sweep (Hausser)
run analysis/figure_2_panel_a_pipeline.py
run analysis/figure_2_panel_b_hull_profile.py
run analysis/figure_2_panel_c_hausser_ablation.py
run analysis/figure_2_panel_d_hausser_hyperparam.py
run analysis/figure_2_supp_nemo.py
run analysis/figure_2_supp_physmap.py
run analysis/figure_2_supp_pipelines.py

# Figure 3 — bimodal benchmark (A1, S1, supp Calvigioni / Ramachandran)
run analysis/figure_3_panel_ab_profile.py
run analysis/figure_3_panel_bimodal_benchmark.py
run analysis/figure_3_panel_c_l4l5.py
run analysis/figure_3_panel_confusion_matrices.py
run analysis/figure_3_supp_isiacg_benchmark.py
run analysis/figure_3_supp_isiacg_profile.py

# Figure 4 — trimodal benchmark (Hull, Lisberger, supp Watson)
run analysis/figure_4_panel_profiles.py
run analysis/figure_4_trimodal_benchmark.py
run analysis/figure_4_supp_trimodal_benchmark.py

# Figure 5 — cross-species benchmark (Lisberger → Hull, four shared types)
run analysis/figure_5_panel_a_method_bars.py
run analysis/figure_5_panel_c_confusion.py
run analysis/figure_5_panel_d_umap.py
run analysis/figure_5_supp_analysis.py
run analysis/figure_5_supp_nopretrain.py

# Figure 6 — generative cVAE (cross-species counterfactual + interpolation)
#
# The G0–G3 scripts re-run the trained-checkpoint analyses (linear probe,
# cross-modal imputation, cross-species decoding, latent interpolation) and
# write CSVs to results/18_cvae_only_experiments/. They require the trained
# checkpoint (results/18_cvae_only_experiments/hippie_cvae.ckpt), the four
# source datasets, and a registry JSON describing per-dataset source_ids.
# Their outputs are already cached in the repo, so only the plotting script
# needs to run for figure regeneration. Reviewers retraining the cVAE should
# invoke G0–G3 manually with --registry / --data-dirs / --datasets.
run analysis/CVAE_only_experiment_plot.py --results-dir results/18_cvae_only_experiments

# Figure 7 — brain region benchmark (Allen + IBL, transductive + holdout)
run analysis/figure_7_brain_region_benchmark.py

# Supplementary figures
run analysis/figure_polarity_supplement.py
# figure_supp_umap_clustering requires the raw C4 database H5 files (~38 GB),
# which exceed GitHub size limits and are not bundled. Skipped here; reviewers
# with the C4 H5 files locally can run it directly.
# run analysis/figure_supp_umap_clustering.py

echo ""
echo "==================== Figure run report ===================="
column -t -s $'\t' "$REPORT"
echo ""
echo "Per-script logs: $LOG_DIR/"
echo "Outputs:         figures/<script-name>/"
