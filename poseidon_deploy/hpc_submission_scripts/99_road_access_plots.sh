#!/bin/bash
# Analysis: plot roadway accessibility statistics across all events.
#
# Input:  $EVENT_DIR, $SITE_DATA/segmentation_labels.csv,
#         $SITE_DATA/sensor_data_comparison.csv
# Output: $SITE_DATA/plots/
#
#BSUB -J plot_access
#BSUB -W 15
#BSUB -n 1
#BSUB -R "rusage[mem=4G]"
#BSUB -q ccee
#BSUB -o job_outputs/poseidon_plots.%J.out
#BSUB -e job_outputs/poseidon_plots.%J.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
activate_env

python -u "$ANALYSIS/access_plotter.py" \
    --site "$SITE" \
    --events_dir "$EVENT_DIR" \
    --out_dir "$SITE_DATA/plots" \
    --labels "$SITE_DATA/segmentation_labels.csv" \
    --sensor "$SITE_DATA/sensor_data_comparison.csv"
