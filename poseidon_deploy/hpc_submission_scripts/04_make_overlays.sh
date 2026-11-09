#!/bin/bash
# Stage 04: draw segmentation overlays on the daylight images for QC.
#
# Runs the compiled overlay_generator on one file list per array task.
# Input:  $SITE_DATA/images/daylight_all_events/ and ..._preds/
# Output: $SITE_DATA/images/daylight_all_events_overlays/
#
#BSUB -J make_overlay[1-32]
#BSUB -W 10
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8]"
#BSUB -q ccee
#BSUB -o job_outputs/make_overlay.%J.%I.out
#BSUB -e job_outputs/make_overlay.%J.%I.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site

IMAGE_DIR="$SITE_DATA/images/daylight_all_events"
FILE_LIST=$(file_list_for_task "$IMAGE_DIR/job_file_lists")
echo "Task $TASK_ID: running overlay_generator on $FILE_LIST"

"$BIN_DIR/overlay_generator" \
    "$IMAGE_DIR" \
    "$SITE_DATA/images/daylight_all_events_preds" \
    "$SITE_DATA/images/daylight_all_events_overlays" \
    "$FILE_LIST" \
    0.6  # overlay opacity
