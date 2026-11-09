#!/bin/bash
# Segmentation: train a Segmentation Gym model.
#
# Edit DATA_DIR and CONFIG below for the dataset and model config.
# Slurm (GPU partition): submit with `sbatch 99_train_model.sh`.
#
#SBATCH --job-name=train_model
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=train_model.%j.out
#SBATCH --error=train_model.%j.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
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
