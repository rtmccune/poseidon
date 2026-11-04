#!/bin/bash
# Stage 02: segment daylight images with the Segmentation Gym container.
#
# One array task per file list; create the lists first with
# 99_create_job_lists.sh and set the array size to their number.
# Input:  $SITE_DATA/images/daylight_all_events/job_file_lists/
# Output: segmentation predictions written by segmentation_gym
#
#BSUB -J seg_batch[1-32]
#BSUB -W 45
#BSUB -n 1
#BSUB -R "rusage[mem=4G]"
#BSUB -gpu "num=1:mode=shared"
#BSUB -q gpu
#BSUB -o job_outputs/seg_batch.%J.%I.out
#BSUB -e job_outputs/seg_batch.%J.%I.err

source "${LS_SUBCWD:-$PWD}/_common.sh"
require_site
module load cuda/12.1
module load apptainer
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_CACHE="$REPO_ROOT/poseidon_deploy/segmentation/segmentation_gym/hf_cache_portable"

IMAGES_DIR="$SITE_DATA/images/daylight_all_events"
WEIGHTS="$REPO_ROOT/data/segmentation/all_sites/weights/all_sites_5_class_v3_segformer_fullmodel.h5"
FILE_LIST=$(file_list_for_task "$IMAGES_DIR/job_file_lists")
echo "Task $LSB_JOBINDEX: $FILE_LIST"

apptainer exec --nv --bind "$REPO_ROOT" \
    "$REPO_ROOT/poseidon_deploy/segmentation/container/seg_gym.sif" \
    python "$REPO_ROOT/poseidon_deploy/segmentation/segmentation_gym/seg_images_in_folder_no_tkinter.py" \
    --images_dir "$IMAGES_DIR" \
    --weights "$WEIGHTS" \
    --file_list "$FILE_LIST"
