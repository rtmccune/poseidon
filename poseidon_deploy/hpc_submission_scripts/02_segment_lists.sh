#!/bin/bash
# Stage 02: segment daylight images with the Segmentation Gym container.
#
# One array task per file list; create the lists first with
# 99_create_job_lists.sh and set the array size to their number.
# Input:  $SITE_DATA/images/daylight_all_events/job_file_lists/
# Output: segmentation predictions written by segmentation_gym
#
# Slurm: submit with `sbatch 02_segment_lists.sh`. --array must match
# the number of file lists. Runs on idle partner RTX PRO 6000 (Blackwell)
# GPUs through the short_gpu QOS, which caps each task at 2 hours. Use a
# different GPU type with e.g. `sbatch --gres=gpu:l40s:1 02_segment_lists.sh`,
# or the standard queue with `--partition=gpu --qos=gpu --gres=gpu:a100:1`.
#
#SBATCH --job-name=seg_batch
#SBATCH --array=1-32
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:rtx_6000_pro:1
#SBATCH --output=job_outputs/seg_batch.%A.%a.out
#SBATCH --error=job_outputs/seg_batch.%A.%a.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
module load cuda/12.1
module load apptainer
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_CACHE="$REPO_ROOT/poseidon_deploy/segmentation/segmentation_gym/hf_cache_portable"

IMAGES_DIR="$SITE_DATA/images/daylight_all_events"
WEIGHTS="$REPO_ROOT/data/segmentation/all_sites/weights/all_sites_5_class_v3_segformer_fullmodel.h5"
FILE_LIST=$(file_list_for_task "$IMAGES_DIR/job_file_lists")
echo "Task $TASK_ID: $FILE_LIST"

apptainer exec --nv --bind "$REPO_ROOT" \
    "$REPO_ROOT/poseidon_deploy/segmentation/container/seg_gym.sif" \
    python "$REPO_ROOT/poseidon_deploy/segmentation/segmentation_gym/seg_images_in_folder_no_tkinter.py" \
    --images_dir "$IMAGES_DIR" \
    --weights "$WEIGHTS" \
    --file_list "$FILE_LIST"
