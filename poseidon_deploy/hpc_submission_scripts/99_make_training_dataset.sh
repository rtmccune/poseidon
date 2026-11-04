#!/bin/bash
# Segmentation: build a Segmentation Gym training dataset from Doodler
# labels.
#
# Edit DATA_DIR and CONFIG below for the dataset to build.
#
#BSUB -J make_data
#BSUB -W 60
#BSUB -n 12
#BSUB -R "rusage[mem=12G]"
#BSUB -gpu "num=1"
#BSUB -q gpu
#BSUB -o make_data.%J.out
#BSUB -e make_data.%J.err

source "${LS_SUBCWD:-$PWD}/_common.sh"
module load cuda/12.1
module load apptainer

DATA_DIR="$REPO_ROOT/data/segmentation/new_bern"  # EDIT
CONFIG="$DATA_DIR/config/new_bern_segformer_v1.json"  # EDIT

apptainer exec --nv --bind "$REPO_ROOT" \
    "$REPO_ROOT/poseidon_deploy/segmentation/container/seg_gym.sif" \
    python "$REPO_ROOT/poseidon_deploy/segmentation/segmentation_gym/make_dataset_no_tkinter_updated.py" \
    --output "$DATA_DIR/fromDoodler/npz4gym" \
    --label_dir "$DATA_DIR/fromDoodler/labels" \
    --image_dirs "$DATA_DIR/fromDoodler/images" \
    --config "$CONFIG"
