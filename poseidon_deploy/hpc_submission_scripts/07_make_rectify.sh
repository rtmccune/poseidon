#!/bin/bash
# Stage 07: rectify each event's images and labels (one task per event).
#
# Slurm (GPU partition): submit with `sbatch 07_make_rectify.sh`.
# Set --array to the number of event folders. To rerun a single event
# (--array=1 on the command line overrides the script's array):
#   TARGET_EVENT=<event folder name> sbatch --array=1 07_make_rectify.sh
# Output: $EVENT_DIR/<event>/zarr/{orig_image_rects,labels_rects}.zip
#
#SBATCH --job-name=rectify_batch
#SBATCH --array=1-19
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=job_outputs/rectify.%A.%a.out
#SBATCH --error=job_outputs/rectify.%A.%a.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
module load cuda/12.6
activate_env

EVENT=$(event_for_task)
echo "Task ${TASK_ID:-single}: $EVENT"

python -u "$NAIADS/run_rectify.py" \
    --site "$SITE" \
    --grid_dir "$REPO_ROOT/data/grids" \
    --target_event_dir "$EVENT"
