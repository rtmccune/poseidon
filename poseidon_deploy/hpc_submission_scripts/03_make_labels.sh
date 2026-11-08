#!/bin/bash
# Stage 03: convert segmentation predictions to grayscale label images.
#
# Runs the compiled pred_label_generator on one file list per array
# task. Create lists of the prediction folder first, e.g.
#   LIST_SOURCE=$SITE_DATA/images/daylight_all_events_preds \
#       bsub < 99_create_job_lists.sh
# Input:  $SITE_DATA/images/daylight_all_events_preds/
# Output: $SITE_DATA/images/daylight_all_events_labels/
#
#BSUB -J make_labels[1-32]
#BSUB -W 10
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4]"
#BSUB -q ccee
#BSUB -o job_outputs/make_labels.%J.%I.out
#BSUB -e job_outputs/make_labels.%J.%I.err

source "${LS_SUBCWD:-$PWD}/_common.sh"
require_site

PREDS_DIR="$SITE_DATA/images/daylight_all_events_preds"
FILE_LIST=$(file_list_for_task "$PREDS_DIR/job_file_lists")
echo "Task $LSB_JOBINDEX: running pred_label_generator on $FILE_LIST"

"$BIN_DIR/pred_label_generator" \
    "$PREDS_DIR" \
    "$FILE_LIST" \
    "$SITE_DATA/images/daylight_all_events_labels"
