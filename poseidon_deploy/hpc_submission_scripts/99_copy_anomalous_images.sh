#!/bin/bash
# Analysis: copy stitched images of flagged anomalies for review.
#
# Run after 99_road_access_plot_outliers.sh and
# 99_image_overlay_comparison.sh.
# Output: $SITE_DATA/images/anomaly_comparisons/<roi>/
#
#BSUB -J copy_imgs
#BSUB -W 15
#BSUB -n 1
#BSUB -R "rusage[mem=4G]"
#BSUB -q ccee
#BSUB -o job_outputs/copy_imgs.%J.out
#BSUB -e job_outputs/copy_imgs.%J.err

source "${LS_SUBCWD:-$PWD}/_common.sh"
require_site
activate_env

python -u "$ANALYSIS/copy_anomalous_images.py" \
    --reports_dir "$SITE_DATA" \
    --images_dir "$SITE_DATA/images/stitched_comparisons" \
    --dest_dir "$SITE_DATA/images/anomaly_comparisons"
