#!/bin/bash
# Segmentation: segment every image in one folder (no job array).
#
# Edit IMAGES_DIR and WEIGHTS below for the folder and model to use.
# Slurm (GPU partition): submit with `sbatch 99_segment_folder.sh`.
#
#SBATCH --job-name=seg_folder
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=seg_folder.%j.out
#SBATCH --error=seg_folder.%j.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
module load cuda/12.1
module load apptainer
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_CACHE="$REPO_ROOT/poseidon_deploy/segmentation/segmentation_gym/hf_cache_portable"

IMAGES_DIR="$REPO_ROOT/data/down_east/images/daylight_all_events"  # EDIT
WEIGHTS="$REPO_ROOT/data/segmentation/all_sites/weights/all_sites_5_class_v3_segformer_fullmodel.h5"  # EDIT

apptainer exec --nv --bind "$REPO_ROOT" \
    "$REPO_ROOT/poseidon_deploy/segmentation/container/seg_gym.sif" \
    python "$REPO_ROOT/poseidon_deploy/segmentation/segmentation_gym/seg_images_in_folder_no_tkinter.py" \
    --images_dir "$IMAGES_DIR" \
    --weights "$WEIGHTS"
