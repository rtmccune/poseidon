#!/bin/bash
# Stage 09: plot depth maps and virtual sensor series (one task per event).
#
# Slurm (CPU partner nodes): submit with `sbatch 09_make_plots.sh`.
# Set --array to the number of event folders. To rerun one event:
#   TARGET_EVENT=<event folder name> sbatch --array=1 09_make_plots.sh
# Runs on idle partner CPU nodes through the short QOS (2-hour limit per
# task). Check a finished task with `seff <jobid>_<task>` and trim
# --time and --mem if they are far above what it used.
#
# The basemap and virtual sensors come from the site config.
# Output: $EVENT_DIR/<event>/plots/
#
#SBATCH --job-name=plot_batch
#SBATCH --array=1-77
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=kanarde_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --output=job_outputs/plotting.%A.%a.out
#SBATCH --error=job_outputs/plotting.%A.%a.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
export TMPDIR=/tmp
export PROJ_NETWORK=OFF  # compute nodes have no internet for PROJ grids
activate_env

EVENT=$(event_for_task)
echo "Task ${TASK_ID:-single}: $EVENT"

python -u "$NAIADS/run_plotter.py" \
    --site "$SITE" \
    --target_event_dir "$EVENT" \
    --stats 95_perc
