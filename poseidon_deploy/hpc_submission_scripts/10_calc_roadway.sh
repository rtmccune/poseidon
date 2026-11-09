#!/bin/bash
# Stage 10: summarize roadway flooding per event (one task per event).
#
# The roadway ROI is the site config's roi_json.
# Output: $EVENT_DIR/<event>/<roi>_{accessibility,wse}_time_series.csv
#
#BSUB -J roadway_batch[1-19]
#BSUB -W 45
#BSUB -n 4
#BSUB -R "rusage[mem=16G]"
#BSUB -q ccee
#BSUB -o job_outputs/roadway.%J.%I.out
#BSUB -e job_outputs/roadway.%J.%I.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
export TMPDIR=/tmp
activate_env

EVENT=$(event_for_task)
echo "Task ${TASK_ID:-single}: $EVENT"

python -u "$NAIADS/run_roadway_analyzer.py" \
    --site "$SITE" \
    --target_event_dir "$EVENT" \
    --label roadway \
    --step_size 1.0 \
    --statistic 95_perc
