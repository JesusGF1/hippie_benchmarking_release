# HIPPIE Benchmark Datasets

All datasets are stored in HIPPIE-compatible CSV format. Each folder contains the
following files (where available):

| File | Shape | Description |
|---|---|---|
| `waveforms.csv` | (N, 46–90) | Normalized spike waveforms (peak channel) |
| `isi_dist.csv` | (N, 100–200) | ISI histograms (1 ms bins, 0–100 ms or 0–200 ms) |
| `acg.csv` | (N, 201) | Autocorrelograms (±100 ms, 1 ms bins) |
| `labels.csv` | (N,) | Cell-type or brain-region labels |
| `metadata.csv` | (N, K) | Per-neuron metadata (session, subject, firing rate, etc.) |
| `splits.json` | — | Predefined train/val/test splits (DANDI datasets only) |
| `class_mapping.json` | — | Label → integer index mapping (DANDI datasets only) |

---

## Quick Reference Table

| Folder | Task | Species | Region | N neurons | N subjects | Cell types / Regions | Paper figures |
|---|---|---|---|---|---|---|---|
| `a1data_remove_undef` | Cell type | Mouse | Auditory cortex (A1) | 285 | — | EXC, PV, SOM | Fig 3 |
| `juxtacellular_mouse_s1_area` | Cell type | Mouse | Somatosensory (S1) | 224 | — | E4, E5, FS4, FS5, SOM | Fig 3 |
| `hausser_cell_type` | Cell type | Mouse | Cerebellar cortex | 1,998 (113 labeled) | — | GoC, GrC, MFB, MLI, PkC\_cs, PkC\_ss | Figs 2, 6 |
| `hull_cell_type` | Cell type | Mouse | Cerebellar cortex | 103 | — | GoC, MFB, MLI, PkC\_cs, PkC\_ss | Fig 4 |
| `lisberger_labeled_cell_type` | Cell type | Macaque | Cerebellar cortex (floccular) | 1,152 (668 labeled) | — | GoC, MFB, MLI, PkC\_cs, PkC\_ss | Figs 4, 5 |
| `cellexplorer_cell_type` | Cell type | Mouse | V1, HVAs, CA1 | 430 | — | PV, SST, VIP, VGAT, Pyramidal, Axo-axonic, Juxtacellular | Hyperparameter tuning only |
| `dandi_000041_cell_type` | Cell type | Rat | Frontal cortex (sleep) | 221 | 9 | Excitatory, Inhibitory | Figs 4 supp, 6 (CVAE) |
| `dandi_000473_cell_type` | Cell type | Mouse | Prefrontal cortex | 9,213 | 25 | Excitatory, Inhibitory | Fig 3 ISI+ACG supp (Calvigioni) |
| `dandi_000955_cell_type` | Cell type | Rat | Somatosensory cortex | 134 | 1 | Excitatory, Inhibitory | Fig 3 ISI+ACG supp (Ramachandran) |
| `allen_scope_neuropixel_area_subset` | Brain region | Mouse | Visual cortex, hippocampus, thalamus, midbrain | 61,781 | 47 | 19 Allen CCF regions | Fig 7 |
| `ibl_brainwide_good` | Brain region | Mouse | Brainwide | 62,993 | 139 | 10 Cosmos regions | Fig 7 |

---

## Cell-Type Classification Datasets

### `a1data_remove_undef` — Extracellular Mouse Auditory Cortex (A1)

**Source:** Lakunina et al. (2020) *Cell Reports* — downloaded from the [PhysMAP GitHub repository](https://github.com/EricKenjiLee/PhysMAP_Manuscript).

**Task:** Cell-type classification (bimodal: waveform + ISI). **Figure 3.**

**Species / Region:** Mouse / Primary auditory cortex (A1)

**Recording technology:** Silicon probes. Optical stimulation via fiber at top of probe recording sites.

**Cell-type identification:** Optogenetic tagging in Pvalb-Cre::Ai32 (PV) and Sst-Cre::Ai32 (SOM) mice. ChR2-expressing cells identified by significant firing-rate increase (p < 0.001) within 10 ms of stimulation. Spikes sorted offline with Klustakwik; units with ISI violation rate > 2% excluded.

**Statistics:**

| Cell type | N |
|---|---|
| EXC (putative excitatory) | 48 |
| PV (Parvalbumin interneurons) | 121 |
| SOM (Somatostatin interneurons) | 116 |
| **Total** | **285** |

**Available modalities:** Waveform, ISI. No ACG (bimodal benchmark only).

---

### `juxtacellular_mouse_s1_area` — Juxtacellular Mouse Somatosensory Cortex (S1)

**Source:** Yu et al. (2019) *Nature Neuroscience* — also known as the "Jianing dataset". Downloaded from the [PhysMAP GitHub repository](https://github.com/EricKenjiLee/PhysMAP_Manuscript).

**Task:** Cell-type classification (bimodal: waveform + ISI). **Figure 3.**

**Species / Region:** Mouse / Primary somatosensory (barrel) cortex

**Recording technology:** Juxtasomal glass micropipettes (*in vivo*). Cells filled post-hoc with biocytin/neurobiotin for morphological verification.

**Cell-type identification:** Transgenic lines Pvalb-Cre::Ai32 (FS) and Sst-Cre::Ai32 (SOM). Spikes from ChR2-expressing neurons identified via laser-evoked responses. Cortical layer inferred from recording depth and morphological alignment.

**Label convention:**

| Label | Meaning |
|---|---|
| `E  4.0` | Excitatory, Layer 4 |
| `E  5.0` | Excitatory, Layer 5 |
| `FS 4.0` | Fast-spiking (PV), Layer 4 |
| `FS 5.0` | Fast-spiking (PV), Layer 5 |
| `SOMnan` | Somatostatin, layer unknown |

**Statistics:** 224 total neurons (58 E4, 43 E5, 35 FS4, 19 FS5, 69 SOM).

**Available modalities:** Waveform, ISI.

---

### `hausser_cell_type` — Hausser Mouse Cerebellar Cortex (C4 Database)

**Source:** Beau et al. (2024) *Nature* — C4 database. Downloaded from <https://www.c4-database.com> (UCL Research Data Repository). H5 file: `C4_database_hausser.h5`.

**Task:** Ablation study and hyperparameter selection. **Figures 2 and 6 (CVAE generative).**

**Species / Region:** Mouse (adult C57BL/6J, >P60) / Cerebellar cortex

**Recording technology:** Neuropixels 1.0 probes, awake head-fixed. Optical stimulation (470 nm) via surface-coupled or tapered fibers.

**Cell-type identification:** Optogenetic tagging via ChR2, GtACR2, or ArchT expressed under cell-type-specific Cre drivers. Confirmed by light-evoked responses ± synaptic blockers (Gabazine, NBQX, APV, MCPG).

**Statistics:**

| Cell type | Labeled |
|---|---|
| GoC (Golgi cells) | 16 |
| GrC (Granule cells) | 9 |
| MFB (Mossy fibers) | 13 |
| MLI (Molecular layer interneurons) | 15 |
| PkC\_cs (Purkinje complex spikes) | 25 |
| PkC\_ss (Purkinje simple spikes) | 35 |
| **Labeled total** | **113** |
| Unlabeled | 1,885 |
| **Dataset total** | **1,998** |

**Role in paper:** This is the hyperparameter tuning dataset. The locked configuration
(z=30, β=1.0) was selected here and frozen before any other benchmark was evaluated.
PhysMAP and NEMO were also swept and locked on this dataset.

**Available modalities:** Waveform, ISI, ACG, spike times.

---

### `hull_cell_type` — Hull Mouse Cerebellar Cortex (C4 Database)

**Source:** Beau et al. (2024) *Nature* — C4 database. H5 file: `C4_database_hull_labelled.h5`. 100% labeled.

**Task:** Trimodal cell-type benchmark. **Figure 4.**

**Species / Region:** Mouse (adult C57BL/6J, >P60) / Cerebellar cortex

**Recording technology:** Neuropixels 1.0 probes, awake head-fixed, same protocol as Hausser dataset.

**Statistics:**

| Cell type | N |
|---|---|
| GoC | 2 |
| MFB | 18 |
| MLI | 13 |
| PkC\_cs | 34 |
| PkC\_ss | 36 |
| **Total** | **103** |

**Available modalities:** Waveform, ISI, ACG, spike times.

---

### `lisberger_labeled_cell_type` — Lisberger Macaque Cerebellar Cortex (C4 Database)

**Source:** Beau et al. (2024) *Nature* — C4 database. H5 file: `C4_database_lisberger.h5`.

**Task:** Trimodal cell-type benchmark and cross-species transfer (macaque→mouse). **Figures 4 and 5.**

**Species / Region:** Adult male rhesus macaque / Cerebellar floccular complex

**Recording technology:** Single tungsten electrodes and 16-channel Plexon s-Probes. Sampling rate: 40 kHz. Signal filtered at 6 kHz.

**Statistics (expert-labeled units):**

| Cell type | N |
|---|---|
| GoC | 188 |
| MFB | 86 |
| MLI | 36 |
| PkC\_cs | 147 |
| PkC\_ss | 211 |
| **Labeled total** | **668** |
| Unlabeled | 484 |
| **Dataset total** | **1,152** |

**Available modalities:** Waveform, ISI, ACG, spike times.

---

### `cellexplorer_cell_type` — CellExplorer Mouse Visual Cortex and Hippocampus

**Source:** Petersen et al. (2021) *Neuron* / Siegle et al. (2021) / Senzai et al. (2019). Downloaded from the [PhysMAP GitHub repository](https://github.com/EricKenjiLee/PhysMAP_Manuscript).

**Task:** Used for hyperparameter tuning in prior versions; not shown in main paper figures.

**Species / Region:** Mouse / Primary visual cortex (V1), higher visual areas (HVAs), hippocampus (CA1)

**Recording technology:** Neuropixels 1.0 probes with optogenetic stimulation.

**Cell-type identification:** Optogenetic tagging via transgenic lines (Pvalb-IRES-Cre::Ai32, Sst-IRES-Cre::Ai32, Vip-IRES-Cre::Ai32). Pyramidal cells identified by waveform shape and firing pattern.

**Statistics:**

| Cell type | N |
|---|---|
| PV | 186 |
| SST | 115 |
| Pyramidal | 44 |
| Axo-axonic | 35 |
| Juxtacellular | 23 |
| VIP | 14 |
| VGAT | 13 |
| **Total** | **430** |

**Available modalities:** Waveform, ISI, ACG.

---

### `dandi_000041_cell_type` — DANDI 000041 Rat Neocortex Sleep

**Source:** [DANDI Archive #000041](https://dandiarchive.org/dandiset/000041), Watson et al. (2016). Silicon-probe recordings from rat frontal cortex during natural sleep.

**Task:** Trimodal supplemental benchmark (Watson) and CVAE cross-dataset generative experiments. **Figures 4 (supp) and 6 (CVAE).**

**Species / Region:** Rat / Frontal cortex (neocortex), natural sleep

**Recording technology:** 64-site silicon probes.

**Cell-type identification:** Pre-labeled in NWB files via `units['cell_type']` column. No optogenetic inference required.

**Statistics:** 221 neurons, 9 subjects, 20 sessions.

| Cell type | N |
|---|---|
| Excitatory | 194 |
| Inhibitory | 27 |
| **Total** | **221** |

**Note:** Original DANDI download ~155 GB. Waveforms stored in `processing/ecephys/SpikeWaveformsX/data` (not the standard `units` table); extracted from peak channel.

**Available modalities:** Waveform, ISI, ACG.

---

### `dandi_000473_cell_type` — Calvigioni (DANDI 000473)

**Source:** [DANDI Archive #000473](https://dandiarchive.org/dandiset/000473), Calvigioni et al. (2023; `calvigioni2023esr1`).

**Task:** Figure 3 ISI+ACG bimodal supplemental benchmark.

**Species / Region:** Mouse / Prefrontal cortex

**Recording technology:** Neuropixels probes.

**Cell-type identification:** Fast Spiking (Inhibitory) vs Regular Spiking (Excitatory), taken from the NWB units table.

**Statistics:** 9,213 neurons, 25 subjects.

| Cell type | N |
|---|---|
| Excitatory (RS) | 7,859 |
| Inhibitory (FS) | 1,354 |
| **Total** | **9,213** |

**Available modalities:** ISI, ACG.

---

### `dandi_000955_cell_type` — Ramachandran (DANDI 000955)

**Source:** [DANDI Archive #000955](https://dandiarchive.org/dandiset/000955), Ramachandran et al. (2022; `ramachandran2022transcranial`).

**Task:** Figure 3 ISI+ACG bimodal supplemental benchmark.

**Species / Region:** Rat / Somatosensory cortex

**Recording technology:** 32-channel NeuroNexus electrodes.

**Cell-type identification:** Excitatory and inhibitory neurons distinguished via CaMKII-targeted opsin expression; only labeled neurons subset from the NWB.

**Statistics:** 134 neurons, 1 subject.

| Cell type | N |
|---|---|
| Inhibitory | 105 |
| Excitatory | 29 |
| **Total** | **134** |

**Available modalities:** ISI, ACG.

---

## Brain Region Datasets

### `allen_scope_neuropixel_area_subset` — Allen Institute Visual Coding Dataset

**Source:** de Vries et al. (2023) / Allen Institute. Accessed via AllenSDK from NWB files; the CSV-conversion steps are summarised in `docs/DATASETS.md` (§ From raw recordings to the bundled CSVs).

**Task:** Brain region classification (transductive and holdout/inductive splits). **Figure 7.**

**Species / Region:** Mouse / Visual cortex, hippocampus, thalamus, midbrain

**Recording technology:** Neuropixels 1.0 probes. Two session types: Brain Observatory 1.1 and Functional Connectivity.

**Quality filters applied:** ISI violations < 0.5, amplitude cutoff < 0.1, presence ratio > 0.9.

**Statistics:** 61,781 units, 47 mice.

| Region | N | Description |
|---|---|---|
| CA1 | 13,451 | Hippocampus CA1 |
| VISal | 5,756 | Visual area anterolateral |
| VISam | 4,922 | Visual area anteromedial |
| VISrl | 4,511 | Visual area rostrolateral |
| DG | 4,300 | Dentate gyrus |
| LP | 4,028 | Lateral posterior thalamus |
| VISp | 7,351 | Primary visual cortex |
| VISl | 3,390 | Visual area lateral |
| VISpm | 3,161 | Visual area posteromedial |
| CA3 | 2,664 | Hippocampus CA3 |
| LGd | 2,293 | Dorsal lateral geniculate |
| SUB | 1,648 | Subiculum |
| ProS | 1,093 | Prosubiculum |
| SGN | 630 | Suprageniculate |
| MGv | 608 | Medial geniculate, ventral |
| VPM | 516 | Ventral posteromedial thalamus |
| PO | 549 | Posterior thalamus |
| Eth | 478 | Ethmoid thalamus |
| MB | 432 | Midbrain |

**Evaluation paradigms:**
- *Transductive:* 80/20 stratified split by brain area
- *Holdout/Inductive:* 37 training mice, 10 held-out animals

**Available modalities:** Waveform, ISI, ACG. `super_regions.csv` contains broad super-region labels (visual cortex, hippocampus, thalamus, midbrain, other) used for Allen region conditioning.

---

### `ibl_brainwide_good` — International Brain Laboratory Brainwide Map (good-QC subset)

**Source:** International Brain Laboratory (IBL). Accessed via ONE API. The dataset retains only units passing IBL's `good` quality label (and the additional filters listed below). Fetch instructions are in `docs/DATASETS.md`.

**Task:** Brain region classification (10 Cosmos regions). **Figure 7.**

**Species / Region:** Mouse / Brainwide (698 probe insertions)

**Recording technology:** Neuropixels probes across 12 laboratories worldwide.

**Quality filters applied:** IBL quality `label == 'good'`, min_spikes ≥ 100, ISI violations < 0.5, presence_ratio ≥ 0.5.

**Statistics:** 62,993 units, 139 subjects, 698 insertions, 12 labs.

**Brain region classes (10 Cosmos-level):**

| Class | Description | N |
|---|---|---|
| isocortex | Neocortex (visual, auditory, somatosensory, motor) | 12,982 |
| TH | Thalamus | 11,124 |
| MB | Midbrain (SC, SNr, VTA) | 8,310 |
| HPF | Hippocampal formation (CA1–CA3, DG, subiculum) | 7,050 |
| HB | Hindbrain (pons, medulla) | 6,950 |
| CNU | Cerebral nuclei (striatum, pallidum) | 6,888 |
| CB | Cerebellum | 5,683 |
| OLF | Olfactory areas | 2,220 |
| CTXsp | Cortical subplate (claustrum) | 1,043 |
| HY | Hypothalamus | 743 |

**Notes:**
- NEMO exhibited numerical instability on this dataset, completing only 1 of 5 folds.

**Available modalities:** Waveform, ISI, ACG.
