#!/bin/bash
# Stage 08: compute water surface and depth maps (one task per event).
#
# Set the array size to the number of event folders. To rerun one
# event: TARGET_EVENT=<event folder name> bsub -J depths_one < 08_calc_depths.sh
# Output: $EVENT_DIR/<event>/zarr/depth_maps.zip and plots/
#
#BSUB -J depths_batch[1-19]
#BSUB -W 45
#BSUB -n 2
#BSUB -R "rusage[mem=8G]"
#BSUB -R "select[a100 || l40 || l40s || h100]"
#BSUB -gpu "num=1:mode=shared"
#BSUB -q gpu
#BSUB -o job_outputs/depths.%J.%I.out
#BSUB -e job_outputs/depths.%J.%I.err

source "${LS_SUBCWD:-$PWD}/_common.sh"
require_site
module load cuda/11.2
activate_env

EVENT=$(event_for_task)
echo "Task ${LSB_JOBINDEX:-single}: $EVENT"

python -u "$NAIADS/run_calc_depths.py" \
    --site "$SITE" \
    --grid_dir "$REPO_ROOT/data/grids" \
    --target_event_dir "$EVENT"
