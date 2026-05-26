# Changelog

All notable changes to this benchmarking release will be documented here.
This project follows [Semantic Versioning](https://semver.org/) and
[Keep a Changelog](https://keepachangelog.com/) conventions.

## [1.0.0] — 2025

### Added

- `examples/smoke_test_hf_checkpoint.py` — one-command sanity check that
  downloads the published `Jesusgf23/hippie` checkpoint and verifies it
  loads with the released `hippie` package and produces a finite latent
  embedding.


Initial release accompanying the HIPPIE manuscript
**A Generative Model for Electrophysiological Analysis Across Species,
Technologies, and Modalities** (currently in peer review; an earlier preprint,
under a different title, is on bioRxiv `10.1101/2025.03.14.642461`).

### Added

- Core HIPPIE package (`hippie/`) — conditional variational autoencoder,
  multimodal data loader, augmentations, and ResNet1D backbones.
- HIPPIE 3D-ACG variant (`hippie_wf3dacg/`) used for the cross-species transfer
  experiment in Figure 5.
- Three baseline implementations under `comparison_methods/`:
  - PhysMAP (R; weighted nearest-neighbor on UMAP-reduced features)
  - NEMO (Python; CLIP-style contrastive learning)
  - WF-RF (handcrafted waveform features + Random Forest)
- Training entry points (`scripts/`) for transductive cross-validation,
  animal-level holdout, and cross-dataset / cross-species transfer.
- Figure-generation scripts (`analysis/`) — one script per main and
  supplementary figure panel, plus the G0–G3 generative-cVAE experiments.
- Processed per-unit CSVs (`datasets/`) for ten benchmark datasets covering
  mouse, rat, and macaque recordings across silicon probes, juxtacellular
  micropipettes, and Neuropixels.
- Cached per-fold predictions (`results/`) sufficient to regenerate every
  paper figure from `bash scripts/run_all_figures.sh` (no GPU, no S3).
- Reproduction documentation (`docs/REPRODUCING.md`,
  `docs/DATASETS.md`) covering install, figure regeneration, per-figure
  training recipes, negative controls, dataset provenance, and the
  preprocessing pipeline that produced the bundled CSVs.
- Quickstart Jupyter notebook (`examples/quickstart.ipynb`) that loads the
  released checkpoint, encodes the bundled datasets, and reproduces the
  G0/G1/G2 generative analyses on CPU in a few minutes.
- Citation metadata (`CITATION.cff`) and pinned R dependency lockfile
  (`comparison_methods/physmap/renv.lock`).

### Notes

- Pretrained HIPPIE checkpoint (`hippie_techcond_v1.ckpt`) is published
  separately on Hugging Face at `Jesusgf23/hippie`.
- The benchmarking code is released under BSD-3-Clause; see `LICENSE`.
