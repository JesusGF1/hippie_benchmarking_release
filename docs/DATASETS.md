# Datasets

Each dataset under `datasets/<name>/` ships with parallel CSV files for the
features it uses: `labels.csv`, `waveforms.csv`, `isi_dist.csv`, and (for the
trimodal benchmarks) `acg.csv`, plus a `metadata.csv` with subject/session
information where available. The bimodal-only benchmarks (A1, juxtacellular
S1) omit `acg.csv`. Raw spike-time pickles have been omitted to keep the
repo small; they are only needed if you wish to recompute the
autocorrelogram at a non-default bin width.

## Datasets included

| Folder                                  | Source                           | Species  | Use                            |
|------------------------------------------|----------------------------------|----------|--------------------------------|
| `a1data_remove_undef/`                  | Lakunina A1                      | mouse    | Fig 3 bimodal benchmark         |
| `cellexplorer_cell_type/`                | Petersen CellExplorer            | mouse    | Fig 6 cross-tech                 |
| `dandi_000041_cell_type/`                | Watson DANDI 000041              | rat      | Fig 4 trimodal supp; Fig 6 generative |
| `dandi_000473_cell_type/`                | Calvigioni DANDI 000473          | mouse    | Fig 3 bimodal supp              |
| `dandi_000955_cell_type/`                | Ramachandran DANDI 000955        | rat      | Fig 3 bimodal supp              |
| `hausser_cell_type/`                    | C4 / Hausser                     | mouse    | Fig 2 (locked HP selection); Fig 4 |
| `hull_cell_type/`                       | C4 / Hull                        | mouse    | Fig 4 trimodal; Fig 5 cross-species target |
| `juxtacellular_mouse_s1_area/`          | Yu juxtacellular S1              | mouse    | Fig 3 bimodal benchmark         |
| `lisberger_labeled_cell_type/`         | C4 / Lisberger                   | macaque  | Fig 4 trimodal; Fig 5 cross-species source |
| `allen_scope_neuropixel_area_subset/`   | Allen Visual Coding              | mouse    | Fig 7 brain region              |

## Datasets fetched on demand

### IBL Brainwide Map (`_good` subset, 62 993 neurons)

The paper uses the IBL quality-filtered "good units" subset. This subset is
too large to ship in the repository; fetch it through the IBL ONE API
(`pip install ONE-api`) or, if you have an S3 mirror, via `aws s3 sync`.

Per-unit selection criteria:

- IBL quality label: `good`
- ISI violations < 0.5
- Presence ratio ≥ 0.5
- Minimum spike count ≥ 100

ONE-API procedure (no S3 access required; the `international` password below
is the documented public read-only credential for the openalyx mirror):

```bash
pip install ONE-api
python - <<'PY'
from one.api import ONE
one = ONE(base_url="https://openalyx.internationalbrainlab.org",
          password="international", silent=True)

# Iterate the Brainwide Map insertions and select units that pass all four
# criteria. For each kept unit, export waveform / ISI / ACG / labels /
# metadata as the per-unit CSVs used by the rest of the pipeline.
PY
```

Place the resulting CSVs under `datasets/ibl_brainwide_good/` with the same
five-file layout (`waveforms.csv`, `isi_dist.csv`, `acg.csv`, `labels.csv`,
`metadata.csv`) used by every other bundled dataset.

For figure regeneration alone this download is unnecessary — the per-fold
cached predictions for Figure 7 are bundled under
`results/benchmark/benchmark_cache/hippie/ibl_brainwide_good/`. The fetch is
only required to retrain HIPPIE on IBL from scratch.

### C4 H5 database (NEMO and WF-RF training)

NEMO and WF-RF train on the C4 database H5 files (one per dataset).
You can download them from the c4 database @ `https://www.c4-database.com/`

**Run NEMO / WF-RF** with `C4_LOCAL_DIR=./datasets/c4_h5/`
(default):
```bash
bash comparison_methods/nemo/scripts/run_nemo_benchmark.sh
python comparison_methods/wf-rf/wf_rf_benchmark_evaluation.py \
    --data-root ./datasets/c4_h5/ --dataset hull_cell_type --cv-fold 0
```

If you mirror the H5 files to an S3 bucket, set `S3_BUCKET` / `S3_PREFIX`
env vars and the NEMO shell script will fetch from there instead.

## Provenance of each dataset

| Dataset   | Source                  | Citation                                        |
|-----------|-------------------------|-------------------------------------------------|
| A1        | PhysMAP GitHub          | Lakunina 2020                                   |
| CellExplorer | PhysMAP GitHub       | Petersen 2021, Siegle 2021                      |
| C4 (Hausser/Hull/Lisberger) | C4 database | Beau 2024                                 |
| DANDI 000041 | DANDI Archive        | Watson 2016                                     |
| DANDI 000473 | DANDI Archive        | Calvigioni 2023                                 |
| DANDI 000955 | DANDI Archive        | Ramachandran 2022                               |
| Allen     | AllenSDK                | Allen Institute Visual Coding                   |
| IBL       | ONE API                 | IBL Brainwide Map                               |
| S1 (Juxtacellular) | PhysMAP GitHub  | Yu 2019                                        |

## File format

All four feature CSVs have one row per unit, in the same row order as
`labels.csv`. The CSVs store each source's **native** feature length; every
modality is resampled to the common length described in Methods § Data
processing at load time (`torch.nn.functional.interpolate`), so the on-disk
column counts vary by source:

- `waveforms.csv`: native peak-channel length (≈ 40–90 columns depending on
  source sampling rate); resampled to 50 samples (2.5 ms @ 20 kHz, trough at 1/3) at load.
- `isi_dist.csv`: native ISI histogram length (≈ 51–200 columns); resampled to
  100 bins (1 ms, 0 – 100 ms) and log(x+1) transformed before normalisation at load.
- `acg.csv`: 201 columns (±100 ms, 1 ms bins); resampled to 100 bins at load.

All three modalities are min-max normalised to `[-1, 1]` at load time. See
`hippie/dataloading.py::MultiModalEphysDataset` for the canonical
preprocessing pipeline.

## From raw recordings to the bundled CSVs

The processed CSVs shipped under `datasets/<name>/` are deterministic outputs
of a per-source feature-extraction pipeline applied to the published raw
data. The pipeline itself is not bundled in this release — running it from
scratch requires the original H5 / NWB / Allen-SDK / ONE-API artefacts (tens
to hundreds of GB) and the per-source SDK environments. This section
documents the operations the pipeline applies so that the bundled CSVs are
fully transparent against the published raw data.

### Common per-modality operations

For every source, each unit's three modalities are computed and saved as one
row of the corresponding CSV in `labels.csv` row order. The operations
applied at extraction time mirror the Methods description:

- **Waveform** — extracted from the peak channel (maximum peak-to-trough
  amplitude), trough-centered with the trough at one-third of the window,
  averaged across 100–500 spikes per unit, and interpolated down to
  50 samples (2.5 ms at 20 kHz) when the source rate is higher.
- **ISI distribution** — binned at 1 ms over a 0–100 ms range
  (100 bins), then log(x + 1) transformed prior to min-max normalisation
  applied at load time.
- **Autocorrelogram (ACG)** — binned at 1 ms over a ±100 ms range
  (201 bins) and resampled to 100 bins via linear interpolation. Omitted
  (no `acg.csv`) only for the bimodal-only A1 and juxtacellular S1 datasets.
  DANDI 000955 (Ramachandran) ships ISI + ACG but no usable waveform channel,
  so its `waveforms.csv` is degenerate and the benchmark uses ISI + ACG only.

All three modalities are min-max normalised to `[-1, 1]` at load time by
`hippie/dataloading.py::MultiModalEphysDataset`, not at extraction time.

### Source-specific quality filters applied at extraction

| Source | Filter |
|---|---|
| Allen Visual Coding | `isi_violations < 0.5`, `amplitude_cutoff < 0.1`, `presence_ratio > 0.9` |
| IBL Brainwide Map (`_good` subset) | IBL quality label `good`, `isi_violations < 0.5`, `presence_ratio >= 0.5`, minimum spike count >= 100 |
| A1 (Lakunina) | Units with ISI violation rate > 2 % excluded |
| C4 (Hausser, Hull, Lisberger) | Use only the expert-labelled subset published with the database |
| DANDI 000041 (Watson), DANDI 000473 (Calvigioni) | Minimum spike count >= 100, `isi_violations <= 0.5`, `presence_ratio >= 0.5`; labels taken from the NWB units table |
| DANDI 000955 (Ramachandran) | Labels taken from the NWB units table; dataset has no waveform channel, so the standard waveform-based QC is not applied (ISI + ACG only) |
| CellExplorer, S1 (juxtacellular) | Use the published label sets from the PhysMAP GitHub mirror as-is |
