#!/bin/bash
# Stage 08: compute water surface and depth maps (one task per event).
#
# Slurm (GPU partition): submit with `sbatch 08_calc_depths.sh`.
# Set --array to the number of event folders. To rerun one event:
#   TARGET_EVENT=<event folder name> sbatch --array=1 08_calc_depths.sh
# Output: $EVENT_DIR/<event>/zarr/depth_maps.zip and plots/
#
#SBATCH --job-name=depths_batch
#SBATCH --array=1-19
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=job_outputs/depths.%A.%a.out
#SBATCH --error=job_outputs/depths.%A.%a.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
module load cuda/11.2
activate_env

EVENT=$(event_for_task)
echo "Task ${TASK_ID:-single}: $EVENT"

python -u "$NAIADS/run_calc_depths.py" \
    --site "$SITE" \
    --grid_dir "$REPO_ROOT/data/grids" \
    --target_event_dir "$EVENT"
