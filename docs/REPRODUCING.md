# Reproducing the paper

This guide walks through reproducing every figure in the manuscript at three
levels of effort: regenerate figures from cached predictions (minutes,
laptop), retrain HIPPIE end-to-end (hours, GPU), or retrain everything
including baselines (days, GPU + R + Python environments).

## 1. Install

```bash
git clone https://github.com/JesusGF1/hippie-benchmarking-release.git
cd hippie-benchmarking-release
conda env create -f environment.yml      # creates env 'hippie'
conda activate hippie
pip install -e .                          # installs hippie package
```

For R-based PhysMAP:

```bash
cd comparison_methods/physmap
R -e 'renv::restore()'                    # restores R deps from renv.lock
```

For NEMO (separate Python env recommended):

```bash
conda env create -f comparison_methods/nemo/environment.yml
```

## 2. Regenerate figures from cached predictions (≈ 5 min)

```bash
make figures
```

Outputs land in `figures/figure_<N>/`. Cached predictions in
`results/benchmark/` and `results/18_cvae_only_experiments/` are exactly the
ones used to produce the published figures.

## 3. Retrain HIPPIE

The training scripts run on a single CUDA-capable server. No cluster
scheduler, no S3, and no Weights & Biases account are required. W&B logging
is opt-in: `wandb.login()` is skipped unless `WANDB_MODE=online` (or
`WANDB_MODE=offline`) is set in the environment.

```bash
# Optional: enable W&B logging (off by default)
# export WANDB_MODE=online && wandb login

# Single fold of a single dataset (sanity check)
python scripts/train_multimodal_transductive.py \
    --dataset hausser_cell_type \
    --cv_fold 0 \
    --output-dir results/hippie_repro/hausser_cell_type/fold_0

# All datasets × 5 folds — loop in shell, e.g.:
for ds in hausser_cell_type hull_cell_type lisberger_labeled_cell_type \
          juxtacellular_mouse_s1_area a1data_remove_undef cellexplorer_cell_type \
          dandi_000041_cell_type dandi_000473_cell_type dandi_000955_cell_type; do
    for fold in 0 1 2 3 4; do
        python scripts/train_multimodal_transductive.py \
            --dataset "$ds" --cv_fold "$fold" \
            --output-dir "results/hippie_repro/$ds/fold_$fold"
    done
done
```

Default training hyperparameters match the locked configuration reported in
Methods § HIPPIE training details:

| Parameter            | Default                              |
|---------------------|---------------------------------------|
| β (KL weight)       | 1.0                                   |
| z (latent dim)      | 30 (= 10 × 3 modalities)              |
| Learning rate       | 1 × 10⁻³                              |
| Batch size          | 128                                   |
| Pretraining epochs  | 100                                   |
| Fine-tuning epochs  | 20                                    |
| Supervised epochs   | 10                                    |
| Config              | `class_decoder_source_bn_aug_reg`     |
| Cross-validation    | Stratified 5-fold (80/20)             |

Modify any of these via CLI flags — see `--help` on each training script.

> **Note:** Fresh retraining numbers may differ from the cached predictions; re-run `bash scripts/run_all_figures.sh` after retraining to obtain consistent figures from your own runs.

## 4. Retrain baselines

```bash
# PhysMAP
bash comparison_methods/physmap/run_physmap.sh

# NEMO  (requires the C4 H5 database; set C4_LOCAL_DIR to the local
#        directory containing the H5 files, or configure S3 — see
#        docs/DATASETS.md)
bash comparison_methods/nemo/scripts/run_nemo_benchmark.sh

# WF-RF
for ds in $(ls datasets); do
    for fold in 0 1 2 3 4; do
        python comparison_methods/wf-rf/wf_rf_benchmark_evaluation.py \
            --dataset-folder datasets/$ds \
            --cv-fold $fold \
            --output-dir results/wf-rf/$ds/fold_$fold
    done
done
```

Each baseline's locked configuration matches the values reported in
Methods § Baselines:

| Method   | Locked config                                    |
|----------|--------------------------------------------------|
| PhysMAP  | dimV = 5, metric = euclidean, nfeatures = 20      |
| NEMO     | τ = 0.5, lr = 1×10⁻⁴, z = 256, batch = 1024,      |
|          |   epochs = 200                                    |
| WF-RF    | RandomForest 200 estimators, max_features='sqrt', |
|          |   9 handcrafted features + PCA ≥ 90 % variance    |

### Reproducing the locked-config selection

The locked configurations for HIPPIE, NEMO, and PhysMAP were picked by the
stability tiebreaker rule described in paper Methods § hyperparameter
parity. The selection is reproducible end-to-end from the bundled sweep
CSV using:

```bash
python scripts/pick_locked_configs.py --verify
```

This reads `results/benchmark/tuning_preview/sweep_results_long.csv`,
applies the 3-level sort (1.96 × SEM tie bucket → lowest std → lowest
numeric config_id as deterministic tiebreak), and checks the result
against the `LOCKED_CONFIG_ID` constants embedded in
`analysis/figure_2_*.py`. All three locked configs reproduce: HIPPIE
`config_25` (z=30, β=1.0), NEMO `config_7` (τ=0.5, lr=1×10⁻⁴, z=256,
batch=1024), and PhysMAP `config_13` (dimV=5, euclidean, n=20).

### Dataset-specific quirks

A few datasets need extra flags to match the paper's class space; the CSV
caches and figure scripts already do the right thing, but if you re-run
the per-method scripts yourself you'll need to pass these explicitly:

| Dataset | Where it matters | Flag |
|---------|------------------|------|
| Hull (Fig 4 trimodal benchmark) | Drops GoC (n = 2 → not a stable test class), giving a **4-class** problem over MFB, MLI, PkC_cs, PkC_ss (chance = 0.25) | NEMO: `--exclude-classes GoC` <br> WF-RF: `--unlabeled-strings "" unlabeled GoC` <br> HIPPIE-WF+3DACG: `--unlabeled-strings "" unlabeled GoC` <br> HIPPIE / PhysMAP: nothing — stratified 5-fold removes GoC from most test folds implicitly |
| Lisberger → Hull (Fig 5 cross-species) | Restricts to the **4 shared cell types** (PkC_ss, PkC_cs, MLI, MFB; chance = 0.25) | No flag needed — `scripts/cross_dataset_script.py` automatically evaluates on the intersection of the train/predict cell types, and Hull's rare GoC (n = 2) drops out under that shared-class restriction. |

Example: re-running the Hull row of Fig 4 from scratch on a server with
the C4 H5 file at `./datasets/hull_cell_type/`:

```bash
# HIPPIE (no flag needed — stratification handles it)
python scripts/train_multimodal_transductive.py \
    --dataset hull_cell_type --cv_fold 0 \
    --output-dir results/hippie/hull_cell_type/fold_0

# NEMO (drop GoC explicitly to match HIPPIE's 4-class test fold)
NEMO_DATASETS_DIR=./datasets/hull_cell_type \
python comparison_methods/nemo/scripts/nemo_benchmark_evaluation.py \
    --dataset-file C4_database_hull_labelled.h5 \
    --out-dir results/nemo/hull_cell_type \
    --cv-fold 0 --n-cv-folds 5 \
    --exclude-classes GoC

# WF-RF (treat GoC as unlabeled to drop it from train+test)
python comparison_methods/wf-rf/wf_rf_benchmark_evaluation.py \
    --data-root ./datasets --dataset hull_cell_type --cv-fold 0 --n-cv-folds 5 \
    --unlabeled-strings "" unlabeled GoC \
    --output-dir results/wf-rf/hull_cell_type/fold_0
```

Forgetting the GoC flag on NEMO / WF-RF will leave GoC in the test fold
whenever stratification puts it there, producing a class space mismatched
to the other methods and inflating per-fold BA variance.

## 5. Per-figure reproduction recipes

Each block below shows the exact training command(s) used to produce the
cached predictions for that figure. Plot scripts read from `results/benchmark/`
and `results/18_cvae_only_experiments/`, so once new predictions are in place
`bash scripts/run_all_figures.sh` regenerates the corresponding figure.

### Figure 2 — Ablation ladder + hyperparameter sweep (Hausser)

```bash
# Ablation rung (one per --config value; see hippie/multimodal_model.py
# CVAEConfig presets for the full list)
for fold in 0 1 2 3 4; do
    python scripts/train_multimodal_transductive.py \
        --dataset hausser_cell_type --cv_fold $fold \
        --config class_decoder_source_bn_aug_reg \
        --output-dir "results/benchmark/figure_2_ablation/$fold"
done

# Hyperparameter sweep cell (one entry of the β × z grid)
python scripts/train_multimodal_transductive.py \
    --dataset hausser_cell_type --cv_fold 0 \
    --beta 1.0 --z_dim 30 \
    --output-dir results/benchmark/figure_2_hpsweep/beta1.0_z30/fold_0
```

Plot scripts: `analysis/figure_2_panel_{a,b,c,d}_*.py`,
`analysis/figure_2_supp_{nemo,physmap,pipelines}.py`.

### Figure 3 — Bimodal cell-type benchmark (waveform + ISI)

```bash
for ds in a1data_remove_undef juxtacellular_mouse_s1_area; do
    for fold in 0 1 2 3 4; do
        python scripts/train_multimodal_transductive_waveisi.py \
            --dataset "$ds" --cv_fold $fold \
            --output-dir "results/benchmark/celltype_cache/hippie-waveisi/$ds/fold_$fold"
    done
done
```

For the ISI+ACG supplement (Calvigioni, Ramachandran):

```bash
for ds in dandi_000473_cell_type dandi_000955_cell_type; do
    for fold in 0 1 2 3 4; do
        python scripts/train_multimodal_transductive_isiacg.py \
            --dataset "$ds" --cv_fold $fold \
            --output-dir "results/benchmark/celltype_cache/hippie-isiacg/$ds/fold_$fold"
    done
done
```

Plot scripts: `analysis/figure_3_*.py`.

### Figure 4 — Trimodal cell-type benchmark

```bash
for ds in hull_cell_type lisberger_labeled_cell_type dandi_000041_cell_type; do
    for fold in 0 1 2 3 4; do
        python scripts/train_multimodal_transductive.py \
            --dataset "$ds" --cv_fold $fold \
            --output-dir "results/benchmark/trimodal_cache/hippie-fullcond/$ds/fold_$fold"
    done
done
```

Plot scripts: `analysis/figure_4_trimodal_benchmark.py`,
`analysis/figure_4_supp_trimodal_benchmark.py`, `analysis/figure_4_panel_profiles.py`.

### Figure 5 — Cross-species transfer (Lisberger ↔ Hull)

```bash
# HIPPIE in both directions, with and without pretraining on other datasets
for direction in \
    "--training_dataset lisberger_labeled_cell_type --predict_dataset hull_cell_type" \
    "--training_dataset hull_cell_type --predict_dataset lisberger_labeled_cell_type"; do
    python scripts/cross_dataset_script.py $direction \
        --config class_decoder_source_bn_aug_reg --beta 1.0 --z_dim 30
done

# HIPPIE+NEMO 3D-ACG variant (one direction shown)
python scripts/train_hippie_wf3dacg_cross_dataset.py \
    --training_dataset lisberger_labeled_cell_type \
    --predict_dataset hull_cell_type
```

Plot scripts: `analysis/figure_5_panel_a_method_bars.py`,
`analysis/figure_5_panel_c_confusion.py`, `analysis/figure_5_panel_d_umap.py`,
`analysis/figure_5_supp_analysis.py`, `analysis/figure_5_supp_nopretrain.py`.

### Figure 6 — Generative cVAE

```bash
# 1. Train the joint-source CVAE checkpoint (multi-dataset)
python analysis/CVAE_only_experiment_build_datasets.py \
    --sources '[{"data_dir":"datasets/hull_cell_type","source_id":0,"super_region_id":0},
                {"data_dir":"datasets/hausser_cell_type","source_id":1,"super_region_id":0},
                {"data_dir":"datasets/dandi_000041_cell_type","source_id":2,"super_region_id":1},
                {"data_dir":"datasets/lisberger_labeled_cell_type","source_id":3,"super_region_id":0}]' \
    --datasets-json results/18_cvae_only_experiments/datasets_train.json \
    --combined-dir results/18_cvae_only_experiments/combined_train

python analysis/CVAE_only_experiment_train_hippie.py \
    --datasets-json results/18_cvae_only_experiments/datasets_train.json \
    --epochs 80 --batch-size 256 --z-dim 16 --beta 0.1 --free-bits 0.5 \
    --out results/18_cvae_only_experiments/hippie_cvae.ckpt

# 2. Run the G0–G3 experiments that produce the published panels
for g in G0_classification_accuracy G1_cross_modal_imputation \
         G2_counterfactual_swap G2_cross_species G3_latent_interpolation; do
    python "analysis/CVAE_only_experiment_$g.py" \
        --results-dir results/18_cvae_only_experiments
done

# 3. Plot
python analysis/CVAE_only_experiment_plot.py \
    --results-dir results/18_cvae_only_experiments
```

### Figure 7 — Brain-region classification (Allen + IBL)

```bash
# Allen — bundled
for fold in 0 1 2 3 4; do
    python scripts/train_multimodal_holdout.py \
        --dataset allen_scope_neuropixel_area_subset \
        --cv_fold $fold \
        --use-region-conditioning \
        --output-dir "results/benchmark/benchmark_cache/hippie-regioncond/allen_scope_neuropixel_area_subset/fold_$fold"
done

# IBL — fetch the good-unit subset first (see docs/DATASETS.md)
for fold in 0 1 2 3 4; do
    python scripts/train_multimodal_holdout.py \
        --dataset ibl_brainwide_good \
        --cv_fold $fold \
        --output-dir "results/benchmark/benchmark_cache/hippie/ibl_brainwide_good/fold_$fold"
done
```

`--use-region-conditioning` passes the Allen super-region (a coarse five-way
grouping of the 19 fine areas) as a conditioning input to the CVAE decoder.
This is used for Allen only; IBL and the cell-type datasets are unconditioned
by design. Drop the flag for the structure-agnostic comparison.

Plot script: `analysis/figure_7_brain_region_benchmark.py`.

## 6. Negative controls

The training scripts implement two control modes used to verify that
HIPPIE's conditioning embeddings carry biological information rather than
acting as a label-leak channel. Both apply only to the training split; the
held-out test split is left untouched.

```bash
# Shuffle: randomly permute training labels (label-leak control)
python scripts/train_multimodal_transductive.py \
    --dataset hausser_cell_type --cv_fold 0 --shuffle_labels \
    --output-dir results/negative_controls/shuffle/hippie/hausser_cell_type/fold_0

# Shift: rotate training labels by one position (class 0 → 1, 1 → 2, …)
python scripts/train_multimodal_transductive.py \
    --dataset hausser_cell_type --cv_fold 0 --shift_labels \
    --output-dir results/negative_controls/shift/hippie/hausser_cell_type/fold_0
```

Bundled `results/negative_controls/` contains pre-computed predictions for
both modes across every dataset.

## 7. Cached results layout

After the figure-regeneration path (`make figures`), the following directories
under `results/` are read by the plot scripts:

| Path | Contents |
|---|---|
| `results/benchmark/celltype_cache/` | Bimodal (Figure 3) per-method × per-dataset × per-fold predictions |
| `results/benchmark/trimodal_cache/` | Trimodal (Figure 4) per-method × per-dataset × per-fold predictions |
| `results/benchmark/cross_dataset_cache/` | Figure 5 cross-species predictions (each method × direction) |
| `results/benchmark/benchmark_cache/` | Figure 7 brain-region predictions (HIPPIE + baselines, Allen + IBL, transductive + holdout) |
| `results/benchmark/figure2_final_ladder.csv` | Figure 2C ablation-ladder summary |
| `results/benchmark/tuning_preview/sweep_results_long.csv` | Figure 2D β × z hyperparameter sweep summary |
| `results/benchmark/brain_region_benchmark.csv` | Figure 7 per-fold metrics across all methods + settings |
| `results/18_cvae_only_experiments/` | Figure 6 generative-cVAE experiments (G0–G3) + trained checkpoint |
| `results/negative_controls/{shuffle,shift}/` | Negative-control predictions for the shuffle/shift label modes |

Per-fold prediction CSVs use two schemas depending on the method:

- Most HIPPIE / NEMO / VAE caches (`celltype_cache/`, `trimodal_cache/`,
  `benchmark_cache/.../{transductive,linear_probe,mlp}_predictions.csv`):
  two columns `pred`, `true` — one row per unit, in dataset row order.
- Cross-dataset caches (`cross_dataset_cache/.../predictions.csv`):
  three columns `predicted_label`, `true_label`, `true_label_original`
  (the last preserves the dataset-native label when classes are remapped).
- PhysMAP holdout caches (`benchmark_cache/physmap-*-holdout/`):
  quoted `"label","prediction","fold"` plus a side-by-side
  `holdout_summary.csv` aggregating per-fold metrics.

The fold index is encoded in the directory path (`.../fold_<k>/`) rather
than in the prediction CSV. Per-fold balanced-accuracy and macro-F1
aggregates live in the summary files at the top of `results/benchmark/`
(`brain_region_benchmark.csv`, `figure2_final_ladder.csv`,
`tuning_preview/sweep_results_long.csv`).

## Troubleshooting

- **GPU memory**: lower `--batch-size` (default 128).
- **PhysMAP R errors**: the bundled `comparison_methods/physmap/renv.lock` pins R 4.2.1 and Seurat 4.3.0 (the script uses Seurat v4 APIs and breaks under Seurat v5). Restore the pinned environment with `R -e 'renv::restore()'` from `comparison_methods/physmap/`, or run `comparison_methods/physmap/setup_script.R` to install the dependency set from scratch.
- **NEMO C4 H5 missing**: download from <https://www.c4-database.com>, place under `./datasets/c4_h5/`, and either set `C4_LOCAL_DIR=./datasets/c4_h5/` before running `run_nemo_benchmark.sh` or pass `--s3-path` if you mirror them to S3.

## Reviewer FAQ

**Q: I don't have AWS / S3 credentials. Can I still run anything?**
Yes — the entire reviewer workflow in `README.md` (install → `make figures`)
works without any S3 access. The training scripts also default to local data:
dataset I/O reads from `./datasets/` (or the path in `HIPPIE_DATA_ROOT` env
var if set), and checkpoint paths only trigger S3 when they start with `s3://`.
The only thing that genuinely needs external fetching is the IBL "good" subset
(Figure 7); for the figure-regeneration path even that is unnecessary because
the per-fold predictions are cached under
`results/benchmark/benchmark_cache/hippie/ibl_brainwide_good/`.

**Q: I don't have a Weights & Biases account. Will the training scripts
prompt me to log in?**
No — wandb is opt-in. The scripts set `WANDB_MODE=disabled` by default; all
`wandb.log()`/`wandb.init()` calls become no-ops, and `wandb.login()` is
skipped. To enable logging, export `WANDB_MODE=online` and run `wandb login`
once before the training script.

**Q: How long does each piece take on a single GPU?**

| Step                                                      | Time     | Hardware           |
|-----------------------------------------------------------|----------|--------------------|
| `make figures` (figure regeneration from cached results)  | ~5 min   | CPU laptop         |
| One HIPPIE training (1 dataset × 1 fold)                  | 2–20 min | 1 NVIDIA GPU       |


Wall-clock varies with GPU model (A100/H100 are faster than
RTX 3090/4090) and with dataset size (Allen and IBL are
substantially slower because of their unit counts).

**Q: I don't have the IBL Brainwide Map locally. Can I still regenerate
Figure 7?**
Yes. The per-fold cached predictions for the IBL panels are bundled under
`results/benchmark/benchmark_cache/hippie/ibl_brainwide_good/`, so
`make figures` regenerates Figure 7 without any IBL data download. You only
need to fetch the raw IBL units (via the ONE API; see `docs/DATASETS.md`) if
you want to *retrain* HIPPIE on IBL from scratch.

**Q: Can I install with bare `pip`, without conda?**
Yes:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
On macOS / Apple Silicon you may need to install `torch` from PyTorch's own
index URL first; see <https://pytorch.org/get-started/locally/>.

**Q: The benchmark scripts try to write to `/data/datasets/`. Is that
required?**
No. `/data/datasets/` is one of several fallback paths probed by
`find_dataset_file()` (originally used by our internal deploy). The same
function also probes `./datasets/`, so a fresh clone with the bundled
datasets just works without any volume mounts. Override with
`HIPPIE_DATA_ROOT=/your/path` if your datasets live elsewhere.

**Q: A figure script fails with `FileNotFoundError` for something under
`results/`. What gives?**
The cached predictions live under `results/benchmark/`. If you have an older
checkout that uses a different layout, update or `git pull`.
