# HIPPIE benchmarking (release)

Repository for reproducing the results reported in
**HIPPIE: A Generative Model for Electrophysiological Analysis Across Species,
Technologies, and Modalities**. Contains HIPPIE's code, the three baseline
methods (PhysMAP, NEMO, WF-RF), and the cached results and processed datasets
used to produce the published figures.

## Repository layout

```
hippie/                      Core CVAE package (HIPPIE main + Unconditioned VAE)
hippie_wf3dacg/                 3D-ACG variant used in Figure 5 (cross-species)
comparison_methods/
  physmap/                   PhysMAP (R, weighted nearest-neighbor)
  nemo/                      NEMO (Python, contrastive)
  wf-rf/                     Handcrafted waveform features + RandomForest
scripts/                     Training entry points (Python)
analysis/                    Figure-generating scripts (one per paper figure)
datasets/                    Processed per-unit CSVs for all benchmarks
results/                     Cached predictions for instant figure regeneration
figures/                     Pre-generated paper figures (PNG + SVG) + regen targets
examples/                    Quickstart notebook (CPU, ~few minutes)
docs/                        Reproduction guide, dataset documentation, model card
```

## Reviewer workflow (no training, ~minutes)

The full sequence to install the environment and regenerate every paper figure
from cached predictions:

```bash
# 1. Install
conda env create -f environment.yml         # creates env named 'hippie'
conda activate hippie
pip install -e .                            # installs the hippie package

# 2. Regenerate every paper figure from cached predictions
bash scripts/run_all_figures.sh             # (or: make figures)
```

The cached predictions in `results/` are exactly the ones used by the
published figures. Step 2 re-runs the plotting scripts only; the resulting
PNGs/SVGs land under `figures/<figure_N>/` next to the pre-generated
versions (modulo matplotlib font-cache nondeterminism). No GPU or S3 access
required.

## Full reproduction from scratch (training, hours of GPU time)

The training scripts run end-to-end on a single Linux server with a CUDA-capable
GPU. No cluster, no S3, and no Weights & Biases account are required —
W&B logging is opt-in via `WANDB_MODE=online` and the dataset I/O reads from
the local `datasets/` directory by default.

```bash
# After the install step above:

# HIPPIE on each dataset × 5 cross-validation folds (one fold shown here).
python scripts/train_multimodal_transductive.py \
    --dataset hausser_cell_type --cv_fold 0
# (Loop over datasets and folds — see docs/REPRODUCING.md)

# Baselines:
bash comparison_methods/physmap/run_physmap.sh
bash comparison_methods/nemo/scripts/run_nemo_benchmark.sh   # set C4_LOCAL_DIR if H5 files live elsewhere
python comparison_methods/wf-rf/wf_rf_benchmark_evaluation.py --help

# Regenerate figures from the new predictions
bash scripts/run_all_figures.sh
```

To enable W&B logging during training, export `WANDB_MODE=online` (and
authenticate once with `wandb login`) before launching the script.

Each script writes its predictions to `results/<method>/<dataset>/fold_<k>/`.
See [docs/REPRODUCING.md](docs/REPRODUCING.md) for the per-figure pipeline
mapping and full CLI reference.

## Datasets

All processed datasets are bundled under `datasets/<name>/`. Each contains:

- `labels.csv` — cell-type or brain-region label per unit
- `waveforms.csv` — mean waveform (50 samples)
- `isi_dist.csv` — inter-spike interval distribution (100 bins)
- `acg.csv` — autocorrelogram (100 bins, ±100 ms) — *trimodal datasets only;
  the bimodal-only benchmarks (A1, juxtacellular S1) omit this file*
- `metadata.csv` — subject/session metadata where available

The IBL Brainwide Map's `_good` quality-filtered subset (62 993 neurons),
used for Figure 7, is fetched on demand because it exceeds GitHub's file
size limits — see [docs/DATASETS.md](docs/DATASETS.md) for the IBL ONE-API
download procedure (no S3 access required).

## Where each figure's cached results live

The bundled predictions under `results/` are organized by experiment, not by
figure number. The table below maps each manuscript figure to the directory
its plot script reads from. The plot scripts themselves live in `analysis/`
and are invoked by `bash scripts/run_all_figures.sh`.

The paths below were verified by tracing each plot script's `open()` calls
during `bash scripts/run_all_figures.sh`. Fig 1 is the manuscript-only
architecture diagram (no plot script in this repo).

| Manuscript figure | What it shows | Cached results path(s) | Plot script(s) |
|---|---|---|---|
| **Fig 2A** | Methodology / pipeline schematic | (no data — pure schematic) | `analysis/figure_2_panel_a_pipeline.py` |
| **Fig 2B** | Hausser dataset profile | (reads `datasets/hausser_cell_type/` directly, not `results/`) | `analysis/figure_2_panel_b_hull_profile.py` *(filename is legacy; renders Hausser)* |
| **Fig 2C** | Ablation ladder (Hausser) | `results/benchmark/figure2_final_ladder.csv`; `results/negative_controls/` | `analysis/figure_2_panel_c_hausser_ablation.py` |
| **Fig 2D** | β × z-dim hyperparameter sweep (Hausser) | `results/benchmark/tuning_preview/sweep_results_long.csv` | `analysis/figure_2_panel_d_hausser_hyperparam.py` |
| **Fig 2 supp NEMO** | NEMO sweep panel | `results/benchmark/tuning_preview/sweep_results_long.csv` | `analysis/figure_2_supp_nemo.py` |
| **Fig 2 supp PhysMAP** | PhysMAP sweep panel | `results/benchmark/tuning_preview/sweep_results_long.csv`; `results/benchmark/celltype_cache/physmap/` | `analysis/figure_2_supp_physmap.py` |
| **Fig 2 supp pipelines** | NEMO/PhysMAP pipeline schematics | (no data — pure schematics) | `analysis/figure_2_supp_pipelines.py` |
| **Fig 3A,B (profile)** | A1 + S1 feature profiles | `results/benchmark/cache_datasets/{a1data_remove_undef,juxtacellular_mouse_s1_area}/` | `analysis/figure_3_panel_ab_profile.py` |
| **Fig 3A,B (bars)** | Bimodal benchmark (A1, S1) | `results/benchmark/celltype_cache/{hippie-waveisi,physmap,vae-waveisi}/<dataset>/`; `results/negative_controls/` | `analysis/figure_3_panel_bimodal_benchmark.py` |
| **Fig 3 confusion** | A1, S1 confusion matrices | `results/benchmark/celltype_cache/{hippie-waveisi,physmap}/<dataset>/` | `analysis/figure_3_panel_confusion_matrices.py` |
| **Fig 3C** | L4 vs L5 misclassification analysis | `results/benchmark/celltype_cache/hippie-waveisi/juxtacellular_mouse_s1_area/`; `results/benchmark/cache_datasets/juxtacellular_mouse_s1_area/` | `analysis/figure_3_panel_c_l4l5.py` |
| **Fig 3 supp profile** | Calvigioni (DANDI 473) + Ramachandran (DANDI 955) profiles | `results/benchmark/cache_datasets/{dandi_000473,dandi_000955}_cell_type/` | `analysis/figure_3_supp_isiacg_profile.py` |
| **Fig 3 supp bars** | ISI+ACG bimodal benchmark | `results/benchmark/celltype_cache/{hippie,physmap}/{dandi_000473,dandi_000955}_cell_type/` | `analysis/figure_3_supp_isiacg_benchmark.py` |
| **Fig 4 profile** | Hull + Lisberger dataset profiles | `results/benchmark/cache_datasets/{hull_cell_type,lisberger_labeled_cell_type}/` | `analysis/figure_4_panel_profiles.py` |
| **Fig 4 bars + CM** | Trimodal benchmark (Hull, Lisberger) | `results/benchmark/trimodal_cache/{hippie-fullcond,nemo,physmap,vae,wf-rf}/<dataset>/`; `results/negative_controls/` | `analysis/figure_4_trimodal_benchmark.py` |
| **Fig 4 supp Watson** | Trimodal benchmark, DANDI 041 (Watson) | `results/benchmark/celltype_cache/{hippie,nemo,physmap}/dandi_000041_cell_type/` | `analysis/figure_4_supp_trimodal_benchmark.py` |
| **Fig 5A,B (bars + metrics)** | Cross-species transfer summary | `results/benchmark/cross_dataset_cache/{hippie-cross-dataset,hippie-cross-dataset-allpretrain,nemo-cross-dataset,nemo-cross-dataset-allpretrain,physmap-cross-dataset,vae-cross-dataset,vae-cross-dataset-allpretrain,hippie-wf3dacg-cross-dataset,hippie-wf3dacg-cross-dataset-allpretrain}/cross_*/` | `analysis/figure_5_panel_a_method_bars.py` |
| **Fig 5C (confusion)** | Cross-species confusion matrices | `results/benchmark/cross_dataset_cache/{hippie-cross-dataset-allpretrain,nemo-cross-dataset-allpretrain,physmap-cross-dataset,vae-cross-dataset}/cross_*/` | `analysis/figure_5_panel_c_confusion.py` |
| **Fig 5D** | Cross-species UMAP | `results/benchmark/cross_dataset_cache/{hippie-cross-dataset-allpretrain,nemo-cross-dataset-allpretrain,physmap-cross-dataset,vae-cross-dataset}/cross_*/predictions/{train,predict}_embeddings.csv` | `analysis/figure_5_panel_d_umap.py` |
| **Fig 5 supp no-pretrain** | No-pretraining ablation | Same nine `cross_dataset_cache/` keys as Fig 5A,B | `analysis/figure_5_supp_nopretrain.py` |
| **Fig 5 supp per-class** | Per-class diagnostics, regioncond variants, reverse direction | `results/benchmark/cross_dataset_cache/{hippie-cross-dataset,hippie-cross-dataset-allpretrain,hippie-cross-dataset-regioncond,hippie-cross-dataset-allpretrain-regioncond,physmap-cross-dataset,vae-cross-dataset,vae-cross-dataset-allpretrain}/cross_*/` | `analysis/figure_5_supp_analysis.py` |
| **Fig 6A,B,C,D** | Generative cVAE (cross-species counterfactual, imputation, linear probe, latent interpolation) | `results/18_cvae_only_experiments/` (`G0_*.csv`, `G1_*.csv`, `G2_*.npz`, `latent_flow.npz`, `hippie_cvae.ckpt`) | `analysis/CVAE_only_experiment_plot.py` |
| **Fig 7** | Brain-region classification (Allen + IBL, transductive + holdout) | `results/benchmark/brain_region_benchmark.csv` (bars); `results/benchmark/benchmark_cache/hippie-regioncond/{allen_scope_neuropixel_area_subset,ibl_brainwide_good}/fold_0/predictions/` (confusion matrices) | `analysis/figure_7_brain_region_benchmark.py` |
| **Polarity supplement** | Within- and between-dataset polarity tables | `results/benchmark/celltype_cache/{hippie,hippie-waveisi}/<dataset>/`; `results/benchmark/trimodal_cache/hippie-fullcond/<dataset>/` | `analysis/figure_polarity_supplement.py` |
| **Negative controls** | Label-shuffle / label-shift sanity checks (consumed by Fig 2C, Fig 3 bars, Fig 4 bars) | `results/negative_controls/{shuffle,shift}/{hippie,hippie-waveisi}/<dataset>/` | (no dedicated plot script; see `docs/REPRODUCING.md` § 6) |

Inside each per-method-per-dataset directory, the standard layout is:

```
<method-key>/<dataset>/fold_<k>/predictions/
    transductive_predictions.csv   # main per-cell predictions (true_label, predicted_label, ...)
    train_embeddings.csv           # for cross-dataset UMAP plots (Fig 5D)
    predict_embeddings.csv         # same, target-side
    timings.csv                    # wall-clock + GPU memory per phase
```

The trained Figure 6 checkpoint (`results/18_cvae_only_experiments/hippie_cvae.ckpt`,
~93 MB) is force-included in the repo so the quickstart notebook works after
a fresh clone. The much larger `Jesusgf23/hippie/hippie_techcond_v1.ckpt`
used by the web app lives on Hugging Face and is downloaded on demand by
`examples/smoke_test_hf_checkpoint.py`.

## Quickstart notebook

`examples/quickstart.ipynb` walks through loading the bundled Figure 6
checkpoint, encoding three benchmark datasets, and reproducing the G0 / G1 / G2
generative analyses on CPU in a few minutes. Open it from the repository root:

```bash
jupyter notebook examples/quickstart.ipynb
```

## Citation

An earlier preprint version of this work is on bioRxiv. The peer-reviewed
manuscript associated with this code release is currently in review and
supersedes the preprint; this section will be updated when the peer-reviewed
version is published. For now, please cite the bioRxiv preprint:

```bibtex
@article{gonzalez-ferrer2025hippie,
  title   = {HIPPIE: A Multimodal Deep Learning Model for Electrophysiological
             Classification of Neurons},
  author  = {Gonzalez-Ferrer, Jesus and Lehrer, Julian and
             Schweiger, Hunter E. and Geng, Jinghui and
             Hernandez, Sebastian and Reyes, Francisco and Sevetson, Jess L. and
             Salama, Sofie R. and Teodorescu, Mircea and
             Haussler, David and Mostajo-Radji, Mohammed A.},
  year    = {2025},
  month   = {3},
  journal = {bioRxiv},
  doi     = {10.1101/2025.03.14.642461},
  url     = {https://www.biorxiv.org/content/10.1101/2025.03.14.642461v1},
  note    = {bioRxiv v1, posted 2025-03-15. The peer-reviewed manuscript
             currently in review has a different title, expanded scope,
             and expanded author list.}
}
```

See `CITATION.cff` for machine-readable citation metadata.

## License

BSD-3-Clause — see [LICENSE](LICENSE).
