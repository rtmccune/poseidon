#!/bin/bash
# Stage 09: plot depth maps and virtual sensor series (one task per event).
#
# The basemap and virtual sensors come from the site config.
# Output: $EVENT_DIR/<event>/plots/
#
#BSUB -J plot_batch[1-77]
#BSUB -W 30
#BSUB -n 4
#BSUB -R "rusage[mem=12G]"
#BSUB -q ccee
#BSUB -o job_outputs/plotting.%J.%I.out
#BSUB -e job_outputs/plotting.%J.%I.err

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
