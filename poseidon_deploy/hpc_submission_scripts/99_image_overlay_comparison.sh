#!/bin/bash
# Utility: stitch each daylight image beside its overlay for review.
#
# Output: $SITE_DATA/images/stitched_comparisons/
#
#BSUB -J stitch
#BSUB -W 60
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8G]"
#BSUB -q ccee
#BSUB -o stitch_images.%J.out
#BSUB -e stitch_images.%J.err

source "${LS_SUBCWD:-$PWD}/_common.sh"
require_site
activate_env

python -u "$ANALYSIS/image_stitcher.py" \
    "$SITE_DATA/images/daylight_all_events" \
    "$SITE_DATA/images/daylight_all_events_overlays" \
    "$SITE_DATA/images/stitched_comparisons" \
    --workers "$LSB_DJOB_NUMPROC"
