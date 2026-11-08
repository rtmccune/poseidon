#!/bin/bash
# Segmentation: segment every image in one folder (no job array).
#
# Edit IMAGES_DIR and WEIGHTS below for the folder and model to use.
#
#BSUB -J seg_folder
#BSUB -W 360
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4G]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[a100 || l40 || l40s || h100]"
#BSUB -q gpu
#BSUB -o seg_folder.%J.out
#BSUB -e seg_folder.%J.err

source "${LS_SUBCWD:-$PWD}/_common.sh"
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
