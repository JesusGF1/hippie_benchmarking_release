#!/usr/bin/env bash
# Example: train a NEMO bimodal embedding on a C4-format dataset and evaluate
# k-NN classification on a held-out labelled set.
#
# Prerequisites:
#   - Install the NEMO package: pip install -e comparison_methods/nemo
#   - C4 H5 datasets available locally (set NEMO_DATASETS_DIR or place files
#     under ./datasets/).
#
# Optional environment variables:
#   export WANDB_ENTITY=<your-entity>
#   export WANDB_PROJECT=nemo_ibl
#   export WANDB_DIR="$HOME/wandb_runs"
#   export NEMO_DATASETS_DIR=/path/to/c4_h5_files

python -m celltype_ibl.models.bimodal_embedding_main_knn \
    --train_dataset_file C4_database_hull_labelled.h5 \
    --test_dataset_file C4_database_lisberger_harmonized_labelled.h5 \
    -e 1000 -k 20 --log_every_n_steps 100 --seed 42 --output_dir ./outputs
