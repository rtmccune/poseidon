#!/bin/bash
# Stage 01: keep images taken between sunrise and sunset at the site.
#
# Input:  $SITE_DATA/images/all_events_during_FOV/
# Output: $SITE_DATA/images/daylight_all_events/
#         $SITE_DATA/abbr_flood_events_daylight.csv (the event sheet plus
#         per-event daylight/night image counts; the original is unchanged)
# The camera latitude/longitude come from the site config.
#
#BSUB -J image_filter
#BSUB -W 15
#BSUB -n 1
#BSUB -q ccee
#BSUB -o image_filter.%J.out
#BSUB -e image_filter.%J.err

source "${LS_SUBCWD:-$PWD}/_common.sh"
require_site
activate_env

python -u "$NAIADS/run_image_filter.py" \
    --site "$SITE" \
    --image_dir "$SITE_DATA/images/all_events_during_FOV" \
    --dest "$SITE_DATA/images/daylight_all_events" \
    --events_csv "$SITE_DATA/abbr_flood_events.csv" \
    --events_out "$SITE_DATA/abbr_flood_events_daylight.csv" \
    --buffer 3  # match --buffer in 00_pull_and_save_images.sh
