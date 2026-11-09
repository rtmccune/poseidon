#!/bin/bash
# Analysis: flag images with unusually low coverage for their depth.
#
# Output: $SITE_DATA/<roi>_anomaly_report.csv and diagnostic plots
#
#BSUB -J anomalies
#BSUB -W 15
#BSUB -n 1
#BSUB -R "rusage[mem=4G]"
#BSUB -q ccee
#BSUB -o job_outputs/anomalies.%J.out
#BSUB -e job_outputs/anomalies.%J.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
activate_env

python -u "$ANALYSIS/find_stat_anomalies.py" \
    --events_dir "$EVENT_DIR" \
    --out_dir "$SITE_DATA"
