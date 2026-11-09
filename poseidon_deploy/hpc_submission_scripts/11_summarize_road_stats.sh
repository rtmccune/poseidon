#!/bin/bash
# Stage 11: summarize roadway accessibility across all events.
#
# Output: summary in the job log and $SITE_DATA/road_access_summary.csv
#
#BSUB -J stats
#BSUB -W 15
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4G]"
#BSUB -q ccee
#BSUB -o summary_stats.%J.out
#BSUB -e summary_stats.%J.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
export TMPDIR=/tmp
activate_env

python -u "$NAIADS/run_road_access_stats.py" \
    --site "$SITE" \
    --event_dir "$EVENT_DIR" \
    --output_csv "$SITE_DATA/road_access_summary.csv"
