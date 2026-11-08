#!/bin/bash
# Utility: split a folder into file lists for stages 02-04.
#
# Lists the daylight images by default; set LIST_SOURCE to list another
# folder (e.g. the predictions folder before stage 03) and NUM_JOBS to
# change the number of lists (default 32).
# Output: <folder>/job_file_lists/file_list_<n>.txt
#
#BSUB -J make_jobs
#BSUB -W 15
#BSUB -n 1
#BSUB -q ccee
#BSUB -o make_jobs.%J.out
#BSUB -e make_jobs.%J.err

source "${LS_SUBCWD:-$PWD}/_common.sh"
require_site
activate_env

SOURCE="${LIST_SOURCE:-$SITE_DATA/images/daylight_all_events}"
python -u "$NAIADS/run_create_file_lists.py" \
    --image_dir "$SOURCE" \
    --output_dir "$SOURCE/job_file_lists" \
    --num_jobs "${NUM_JOBS:-32}"
