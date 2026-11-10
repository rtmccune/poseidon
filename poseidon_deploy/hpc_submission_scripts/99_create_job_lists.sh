#!/bin/bash
# Utility: split a folder into file lists for stages 02-04.
#
# Lists the daylight images by default; set LIST_SOURCE to list another
# folder (e.g. the predictions folder before stage 03) and NUM_JOBS to
# change the number of lists (default 32).
#
# To list only the images not yet segmented (e.g. to finish stage 02 on
# CPU after part of it ran on GPU), set SKIP_DONE=1. The lists then go to
# job_file_lists_remaining/ so lists still in use are not overwritten:
#   SKIP_DONE=1 NUM_JOBS=100 sbatch 99_create_job_lists.sh
#   LIST_DIR=job_file_lists_remaining sbatch --array=1-<n> 02_segment_lists_cpu.sh
# The log gives <n>, the number of lists written.
# Output: <folder>/job_file_lists/file_list_<n>.txt
#
#BSUB -J make_jobs
#BSUB -W 15
#BSUB -n 1
#BSUB -q ccee
#BSUB -o make_jobs.%J.out
#BSUB -e make_jobs.%J.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
activate_env

SOURCE="${LIST_SOURCE:-$SITE_DATA/images/daylight_all_events}"
EXTRA=()
LIST_DIR="job_file_lists"
if [ -n "$SKIP_DONE" ]; then
    EXTRA=(--skip_done_dir "$SITE_DATA/images/daylight_all_events_preds")
    LIST_DIR="job_file_lists_remaining"
fi
python -u "$NAIADS/run_create_file_lists.py" \
    --image_dir "$SOURCE" \
    --output_dir "$SOURCE/$LIST_DIR" \
    --num_jobs "${NUM_JOBS:-32}" \
    "${EXTRA[@]}"
