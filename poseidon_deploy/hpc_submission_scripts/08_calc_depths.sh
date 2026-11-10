#!/bin/bash
# Stage 08: compute water surface and depth maps (one task per event).
#
# Slurm (CPU partner nodes): submit with `sbatch 08_calc_depths.sh`.
# Set --array to the number of event folders. To rerun one event:
#   TARGET_EVENT=<event folder name> sbatch --array=1 08_calc_depths.sh
# Output: $EVENT_DIR/<event>/zarr/depth_maps.zip and plots/
#
# Runs on the CPU: these steps are limited by file reads and writes, not
# computation (an event takes 1-7 minutes, using ~1 GB and well under
# one core), so a GPU does not help. Uses idle partner CPU nodes through
# the short QOS (2-hour limit per task).
#
#SBATCH --job-name=depths_batch
#SBATCH --array=1-82
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=kanarde_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --output=job_outputs/depths.%A.%a.out
#SBATCH --error=job_outputs/depths.%A.%a.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
activate_env

EVENT=$(event_for_task)
echo "Task ${TASK_ID:-single}: $EVENT"

python -u "$NAIADS/run_calc_depths.py" \
    --site "$SITE" \
    --grid_dir "$REPO_ROOT/data/grids" \
    --target_event_dir "$EVENT" \
    --disable_gpu
