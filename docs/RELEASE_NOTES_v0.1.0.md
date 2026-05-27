# HIPPIE benchmarking — v0.1.0

Initial public release of the benchmarking codebase accompanying the HIPPIE
manuscript **A Generative Model for Electrophysiological Analysis Across
Species, Technologies, and Modalities**, currently in peer review. An earlier
preprint — under a different title — is on
[bioRxiv](https://www.biorxiv.org/content/10.1101/2025.03.14.642461v1); see the
citation below.

## Contents

- Core HIPPIE package and the 3D-ACG variant used for cross-species transfer
- Three baseline implementations (PhysMAP, NEMO, WF-RF) with locked configurations
- Ten processed per-unit datasets (mouse, rat, macaque; silicon probes,
  juxtacellular, Neuropixels)
- Cached per-fold predictions sufficient to regenerate every paper figure
- Per-figure reproduction recipes and the preprocessing pipeline description
- The pretrained checkpoint `hippie_techcond_v1.ckpt` is published separately
  at [Hugging Face: `Jesusgf23/hippie`](https://huggingface.co/Jesusgf23/hippie)
  under Apache-2.0
- Tutorial notebooks (load the checkpoint, embed/classify, train on your own
  data) live in the HIPPIE package repo, `braingeneers/HIPPIE` (`examples/`)

## Reviewer workflow

```bash
git clone https://github.com/JesusGF1/hippie-benchmarking-release.git
cd hippie-benchmarking-release
conda env create -f environment.yml
conda activate hippie
pip install -e .
make figures
```

Outputs land in `figures/figure_<N>/` and match the bundled reference images
modulo matplotlib font-cache nondeterminism. No GPU, S3, or W&B account
required.

See `docs/REPRODUCING.md` for the per-figure retrain recipes and
`docs/DATASETS.md` for dataset provenance and the preprocessing pipeline.

## Cite

An earlier preprint of this work is on bioRxiv (cited below); the
peer-reviewed manuscript associated with this release is currently in review
and has a different title, expanded scope, and expanded author list. Cite the
bioRxiv preprint for now:

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

## License

- Benchmarking code: BSD-3-Clause (`LICENSE`)
- Pretrained checkpoint on Hugging Face: Apache-2.0
