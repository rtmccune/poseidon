#!/bin/bash
# Segmentation: build a Segmentation Gym training dataset from Doodler
# labels.
#
# Edit DATA_DIR and CONFIG below for the dataset to build.
# Slurm (GPU partition): submit with `sbatch 99_make_training_dataset.sh`.
#
#SBATCH --job-name=make_data
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=12G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=make_data.%j.out
#SBATCH --error=make_data.%j.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
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
