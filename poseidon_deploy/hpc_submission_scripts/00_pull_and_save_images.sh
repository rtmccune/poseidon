#!/bin/bash
# Stage 00: copy archived images taken during each flood event.
#
# Input:  $SITE_DATA/abbr_flood_events.csv; archive at $IMAGE_DRIVE
# Output: $SITE_DATA/images/all_events_during_FOV/
#
#BSUB -J image_pull
#BSUB -W 10
#BSUB -n 1
#BSUB -q ccee
#BSUB -o image_pull.%J.out
#BSUB -e image_pull.%J.err

source "${LS_SUBCWD:-$PWD}/_common.sh"
require_site
[ -n "$IMAGE_DRIVE" ] || die "IMAGE_DRIVE is not set in hpc_paths.env"
activate_env

python -u "$NAIADS/run_image_pull.py" \
    --drive "$IMAGE_DRIVE" \
    --csv "$SITE_DATA/abbr_flood_events.csv" \
    --dest "$SITE_DATA/images/all_events_during_FOV" \
    --buffer 3
