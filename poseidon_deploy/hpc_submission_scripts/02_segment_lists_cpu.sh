#!/bin/bash
# Stage 02 (CPU): segment daylight images without a GPU.
#
# Same work as 02_segment_lists.sh, but on CPU nodes, which are not
# limited by the per-user GPU cap of the short_gpu QOS. Prediction is a
# small part of each image's time (the model runs at 512 x 512); most of
# it is reading the full-size image and writing the full-size outputs,
# which is CPU work either way.
#
# Test one list first and compare its time with a GPU task's log:
#   sbatch --array=1 02_segment_lists_cpu.sh
# Then submit the rest, with --array matching the number of file lists.
# More, smaller lists run more tasks at once, e.g.
#   NUM_JOBS=200 sbatch 99_create_job_lists.sh
#   sbatch --array=1-200 02_segment_lists_cpu.sh
# Each task must finish within the short QOS time limit.
#
# Input:  $SITE_DATA/images/daylight_all_events/job_file_lists/
# Output: segmentation predictions written by segmentation_gym
#
#SBATCH --job-name=seg_cpu
#SBATCH --array=1-32
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --account=kanarde_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --output=job_outputs/seg_cpu.%A.%a.out
#SBATCH --error=job_outputs/seg_cpu.%A.%a.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
module load apptainer
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_CACHE="$REPO_ROOT/poseidon_deploy/segmentation/segmentation_gym/hf_cache_portable"
# Hide any GPU and limit TensorFlow to the CPUs given to the task
export APPTAINERENV_CUDA_VISIBLE_DEVICES=""
export APPTAINERENV_OMP_NUM_THREADS="$NPROC"
export APPTAINERENV_TF_NUM_INTRAOP_THREADS="$NPROC"
export APPTAINERENV_TF_NUM_INTEROP_THREADS=2

IMAGES_DIR="$SITE_DATA/images/daylight_all_events"
WEIGHTS="$REPO_ROOT/data/segmentation/all_sites/weights/all_sites_5_class_v3_segformer_fullmodel.h5"
FILE_LIST=$(file_list_for_task "$IMAGES_DIR/job_file_lists")
echo "Task $TASK_ID: $FILE_LIST on $NPROC CPUs"

SECONDS=0
apptainer exec --bind "$REPO_ROOT" \
    "$REPO_ROOT/poseidon_deploy/segmentation/container/seg_gym.sif" \
    python "$REPO_ROOT/poseidon_deploy/segmentation/segmentation_gym/seg_images_in_folder_no_tkinter.py" \
    --images_dir "$IMAGES_DIR" \
    --weights "$WEIGHTS" \
    --file_list "$FILE_LIST"
echo "Task $TASK_ID: $(wc -l < "$FILE_LIST") images in $((SECONDS / 60)) min"
