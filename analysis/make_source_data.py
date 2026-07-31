"""Build the Nature Communications Source Data workbook.

Writes one sheet per manuscript panel containing the values plotted in that
panel. Nothing is recomputed from raw recordings: every sheet is a view of the
cached per-fold predictions and metrics that produced the figures.

Sheets sourced from ``figures/**/*_metrics.csv`` require those tables to exist,
so run the corresponding figure scripts first (or ``make figures``):

    python analysis/figure_3_panel_bimodal_benchmark.py
    python analysis/figure_3_panel_c_l4l5.py
    python analysis/figure_3_supp_isiacg_benchmark.py
    python analysis/figure_4_trimodal_benchmark.py
    python analysis/figure_4_supp_trimodal_benchmark.py

Usage:

    python analysis/make_source_data.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results" / "benchmark"
FIGS = REPO_ROOT / "figures"
DEFAULT_OUT = FIGS / "Source_Data.xlsx"

# Dataset key -> name used in the manuscript
DATASET_NAME = {
    "hausser_cell_type": "Hausser (mouse cerebellar cortex)",
    "hull_cell_type": "Hull (mouse cerebellar cortex)",
    "lisberger_labeled_cell_type": "Lisberger (macaque cerebellar cortex)",
    "a1data_remove_undef": "Extracellular Mouse A1",
    "juxtacellular_mouse_s1_area": "Juxtacellular Mouse S1",
    "dandi_000041_cell_type": "Watson (rat neocortex)",
    "dandi_000473_cell_type": "Calvigioni (mouse prefrontal cortex)",
    "dandi_000955_cell_type": "Ramachandran (rat somatosensory cortex)",
    "cellexplorer_cell_type": "CellExplorer",
    "allen_scope_neuropixel_area_subset": "Allen Institute Visual Coding",
    "ibl_brainwide_good": "IBL Brainwide Map",
}


def build() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    sheets: dict[str, pd.DataFrame] = {}
    index_rows: list[dict[str, str]] = []

    def add(panel: str, description: str, df: pd.DataFrame, source: str) -> None:
        """Register one sheet; sheet names are capped at Excel's 31-char limit."""
        name = panel[:31]
        df = df.copy()
        if "dataset" in df.columns:
            df.insert(
                df.columns.get_loc("dataset") + 1,
                "dataset_name",
                df["dataset"].map(DATASET_NAME).fillna(df["dataset"]),
            )
        sheets[name] = df
        index_rows.append({
            "Sheet": name,
            "Panel": panel,
            "Description": description,
            "Rows": str(len(df)),
            "Source file (hippie_benchmarking_release)": source,
        })

    # --- Figure 2c: ablation ladder ---------------------------------------
    ladder = pd.read_csv(RESULTS / "figure2_final_ladder.csv")
    ladder = ladder[ladder["dataset"] == "hausser_cell_type"]
    add(
        "Figure 2c",
        "Balanced accuracy per cross-validation fold for each of the eight "
        "architectural variants in the ablation ladder. Bars in the figure are "
        "the mean over the 5 folds; error bars are +/-1 SD over the same "
        "5 values.",
        ladder,
        "results/benchmark/figure2_final_ladder.csv",
    )

    # --- Figure 2d / 2e and the method sweeps -----------------------------
    sweep = pd.read_csv(RESULTS / "tuning_preview" / "sweep_results_long.csv")
    locked = json.loads(
        (RESULTS / "tuning_preview" / "locked_configs.json").read_text()
    )

    hippie_knn = sweep[(sweep["method"] == "hippie") & (sweep["classifier"] == "knn")]
    add(
        "Figure 2d",
        "Balanced accuracy per fold for every (latent dimensionality z_dim x "
        "beta) configuration of the HIPPIE hyperparameter sweep, KNN probe. "
        "Each cell of the grid in the figure is the mean over the 5 folds.",
        hippie_knn[["config_id", "z_dim", "beta", "dataset", "fold",
                    "balanced_accuracy"]],
        "results/benchmark/tuning_preview/sweep_results_long.csv",
    )

    locked_hippie = locked["configs"]["hippie"]["config_id"]
    add(
        "Figure 2e",
        f"KNN vs MLP classification head at the locked HIPPIE configuration "
        f"({locked_hippie}), balanced accuracy per fold.",
        sweep[(sweep["method"] == "hippie") & (sweep["config_id"] == locked_hippie)][
            ["config_id", "classifier", "z_dim", "beta", "dataset", "fold",
             "balanced_accuracy"]
        ],
        "results/benchmark/tuning_preview/sweep_results_long.csv",
    )

    # --- Figures 3a-b: bimodal cell-type benchmarks -----------------------
    bimodal = pd.read_csv(FIGS / "figure_3" / "bimodal_metrics.csv")
    for panel, key in (("Figure 3a", "a1data_remove_undef"),
                       ("Figure 3b", "juxtacellular_mouse_s1_area")):
        add(
            panel,
            f"Balanced accuracy and macro F1 per fold, per method, on the "
            f"{DATASET_NAME[key]} dataset.",
            bimodal[bimodal["dataset"] == key],
            "figures/figure_3/bimodal_metrics.csv",
        )

    # --- Figure 3c: Layer 4 / Layer 5 counts ------------------------------
    add(
        "Figure 3c",
        "Counts of correctly classified and misclassified Layer 4 / Layer 5 "
        "excitatory neurons, pooled across the 5 cross-validation test folds.",
        pd.read_csv(FIGS / "figure_3" / "l4l5_counts.csv"),
        "figures/figure_3/l4l5_counts.csv",
    )

    # --- Figures 4b, 4e: trimodal benchmarks ------------------------------
    trimodal = pd.read_csv(FIGS / "figure_4" / "trimodal_metrics.csv")
    for panel, key in (("Figure 4b", "hull_cell_type"),
                       ("Figure 4e", "lisberger_labeled_cell_type")):
        add(
            panel,
            f"Balanced accuracy and macro F1 per fold, per method, on the "
            f"{DATASET_NAME[key]} dataset.",
            trimodal[trimodal["dataset"] == key],
            "figures/figure_4/trimodal_metrics.csv",
        )

    # --- Figure 7c: brain-region benchmark --------------------------------
    add(
        "Figure 7c",
        "Balanced accuracy and macro F1 per fold, per method, for the Allen and "
        "IBL brain-region benchmarks in both the transductive and animal-holdout "
        "splits. n_test is the number of held-out units contributing to that "
        "fold.",
        pd.read_csv(RESULTS / "brain_region_benchmark.csv"),
        "results/benchmark/brain_region_benchmark.csv",
    )

    # --- Supplementary sheets ---------------------------------------------
    nemo = sweep[sweep["method"] == "nemo"]
    add(
        "Supplementary Figure 1b-c",
        "Balanced accuracy per fold for all 42 NEMO hyperparameter "
        "configurations on the Hausser dataset, for both the KNN and MLP probes.",
        nemo[["config_id", "classifier", "temperature", "lr", "batch_size",
              "weight_decay", "dataset", "fold", "balanced_accuracy"]],
        "results/benchmark/tuning_preview/sweep_results_long.csv",
    )

    physmap = sweep[sweep["method"] == "physmap"]
    add(
        "Supplementary PhysMAP sweep",
        "Balanced accuracy for all 36 PhysMAP configurations (reduced-dimension "
        "size dimV x number of features nfeatures) on the Hausser dataset.",
        physmap[["config_id", "classifier", "dimV", "nfeatures", "metric",
                 "dataset", "fold", "balanced_accuracy"]],
        "results/benchmark/tuning_preview/sweep_results_long.csv",
    )

    add(
        "Supplementary ISI-ACG benchmark",
        "Balanced accuracy and macro F1 per fold, per method, for the ISI+ACG "
        "bimodal benchmarks on the Calvigioni and Ramachandran datasets.",
        pd.read_csv(FIGS / "figure_3" / "supplement" / "isiacg_metrics.csv"),
        "figures/figure_3/supplement/isiacg_metrics.csv",
    )

    add(
        "Supplementary Watson benchmark",
        "Balanced accuracy and macro F1 per fold, per method, for the trimodal "
        "benchmark on the Watson dataset.",
        pd.read_csv(FIGS / "figure_4" / "supplement" / "supp_trimodal_metrics.csv"),
        "figures/figure_4/supplement/supp_trimodal_metrics.csv",
    )

    return sheets, pd.DataFrame(index_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output .xlsx path (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    sheets, index = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        index.to_excel(writer, sheet_name="Index", index=False)
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

        for ws in writer.book.worksheets:
            for column in ws.columns:
                width = max(len(str(c.value)) for c in column if c.value is not None)
                ws.column_dimensions[column[0].column_letter].width = min(
                    max(width + 2, 10), 60
                )

    print(f"[xlsx] {args.out}")
    for _, row in index.iterrows():
        print(f"  {row['Sheet']:34s} {row['Rows']:>5s} rows")


if __name__ == "__main__":
    main()
