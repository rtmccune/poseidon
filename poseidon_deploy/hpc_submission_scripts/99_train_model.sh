#!/bin/bash
# Segmentation: train a Segmentation Gym model.
#
# Edit DATA_DIR and CONFIG below for the dataset and model config.
#
#BSUB -J train_model
#BSUB -W 60
#BSUB -n 12
#BSUB -R "rusage[mem=10G]"
#BSUB -R "select[a100 || l40 || l40s || h100]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpu
#BSUB -o train_model.%J.out
#BSUB -e train_model.%J.err

source "${LS_SUBCWD:-$PWD}/_common.sh"
module load cuda/12.1
module load apptainer
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_CACHE="$REPO_ROOT/poseidon_deploy/segmentation/segmentation_gym/hf_cache_portable"

DATA_DIR="$REPO_ROOT/data/segmentation/new_bern"  # EDIT
CONFIG="$DATA_DIR/config/new_bern_segformer_v1.json"  # EDIT
NPZ_DIR="$DATA_DIR/fromDoodler/npz4gym"

apptainer exec --nv --bind "$REPO_ROOT" \
    "$REPO_ROOT/poseidon_deploy/segmentation/container/seg_gym.sif" \
    python "$REPO_ROOT/poseidon_deploy/segmentation/segmentation_gym/train_model_script_no_tkinter.py" \
    --train_data_dir "$NPZ_DIR/train_data/train_npzs" \
    --val_data_dir "$NPZ_DIR/val_data/val_npzs" \
    --config_file "$CONFIG"
