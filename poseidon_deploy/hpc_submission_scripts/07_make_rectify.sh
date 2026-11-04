#!/bin/bash
# Stage 07: rectify each event's images and labels (one task per event).
#
# Set the array size to the number of event folders. To rerun a single
# event (-J overrides the array):
#   TARGET_EVENT=<event folder name> bsub -J rectify_one < 07_make_rectify.sh
# Output: $EVENT_DIR/<event>/zarr/{orig_image_rects,labels_rects}.zip
#
#BSUB -J rectify_batch[1-19]
#BSUB -W 10
#BSUB -n 4
#BSUB -R "rusage[mem=4G]"
#BSUB -R "select[a100 || l40 || l40s || h100]"
#BSUB -gpu "num=1:mode=shared"
#BSUB -q gpu
#BSUB -o job_outputs/rectify.%J.%I.out
#BSUB -e job_outputs/rectify.%J.%I.err

source "${LS_SUBCWD:-$PWD}/_common.sh"
require_site
module load cuda/12.6
activate_env

EVENT=$(event_for_task)
echo "Task ${LSB_JOBINDEX:-single}: $EVENT"

python -u "$NAIADS/run_rectify.py" \
    --site "$SITE" \
    --grid_dir "$REPO_ROOT/data/grids" \
    --target_event_dir "$EVENT"
