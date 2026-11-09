#!/bin/bash
# Stage 05: create one folder per flood event with its data and images.
#
# Input:  $SITE_DATA/abbr_flood_events.csv, flood_events.csv, and the
#         daylight images and labels
# Output: $EVENT_DIR/<event>/ with <event>.csv, orig_images/, labels/
#
#BSUB -J gen_flood_folders
#BSUB -W 15
#BSUB -n 1
#BSUB -q ccee
#BSUB -o gen_flood_folders.%J.out
#BSUB -e gen_flood_folders.%J.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
activate_env

python -u "$NAIADS/run_flood_event_setup.py" \
    --abbr_events_csv "$SITE_DATA/abbr_flood_events.csv" \
    --full_sensor_csv "$SITE_DATA/flood_events.csv" \
    --image_dir "$SITE_DATA/images/daylight_all_events" \
    --label_dir "$SITE_DATA/images/daylight_all_events_labels" \
    --output_dir "$EVENT_DIR" \
    --image_subfolder orig_images \
    --label_subfolder labels \
    --padding_hours 3
