#!/bin/bash

# Labeled datasets to test on "C4_database_hull_labelled.h5"
LABELED_DATASETS=(
    "C4_database_lisberger_harmonized_labelled.h5"
)
#"C4_database_hausser.h5"
# All available datasets for pretraining
ALL_PRETRAIN_DATASETS=(
    "C4_database_hull_unlabelled.h5"
    "C4_database_hull_labelled.h5"
    "C4_database_hausser.h5"
    "C4_database_medina_unlabelled.h5"
    "C4_database_lisberger_harmonized_labelled.h5"
)

# Number of CV folds
N_FOLDS=5

# Training parameters
Z_DIM=512
PRETRAIN_EPOCHS=5000
BATCH_SIZE=1024
LEARNING_RATE=0.0001

# Loop through each labeled dataset
for DATASET in "${LABELED_DATASETS[@]}"; do
    echo "=========================================="
    echo "Running transductive test on: $DATASET"
    echo "=========================================="

    # Build pretrain datasets list excluding the current test dataset
    PRETRAIN_DATASETS=""
    for PRETRAIN_DS in "${ALL_PRETRAIN_DATASETS[@]}"; do
        if [ "$PRETRAIN_DS" != "$DATASET" ]; then
            PRETRAIN_DATASETS="$PRETRAIN_DATASETS $PRETRAIN_DS"
        fi
    done

    echo "Pretraining datasets: $PRETRAIN_DATASETS"
    echo ""

    # Loop through each CV fold
    for FOLD in $(seq 0 $((N_FOLDS-1))); do
        echo "Running fold $FOLD/$N_FOLDS for $DATASET..."

        python NEMO_ICLR/scripts/nemo_transductive_test.py \
            --dataset-file "$DATASET" \
            --pretrain-datasets $PRETRAIN_DATASETS \
            --cv_fold $FOLD \
            --n_cv_folds $N_FOLDS \
            --z_dim $Z_DIM \
            --pretrain-max-epochs $PRETRAIN_EPOCHS \
            --batch-size $BATCH_SIZE \
            --learning-rate $LEARNING_RATE \
            --project NEMO_transductive \
            --output-dir ./outputs_nemo_transductive

        if [ $? -ne 0 ]; then
            echo "Error running fold $FOLD for $DATASET"
            exit 1
        fi

        echo "Completed fold $FOLD for $DATASET"
    done

    echo ""
    echo "Completed all folds for $DATASET"
    echo ""
done

echo "=========================================="
echo "All transductive tests completed!"
echo "=========================================="