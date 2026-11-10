#!/bin/bash
# Stage 04: draw segmentation overlays on the daylight images for QC.
#
# Runs the compiled overlay_generator on one file list per array task,
# using the image lists made for stage 02 (99_create_job_lists.sh).
# Submit with `sbatch 04_make_overlays.sh`; --array must match the
# number of lists (NUM_JOBS, default 32). Can run alongside stage 03.
#
# Each image is read, blended with its prediction, and written as a
# JPEG, so a list of ~160 images takes a few minutes on 8 CPUs. The tool
# starts one thread per CPU given to the task (POSEIDON_THREADS);
# rebuild it (README: C++ Executables) if its log says it is using every
# core on the node.
#
# Input:  $SITE_DATA/images/daylight_all_events/ and ..._preds/
# Output: $SITE_DATA/images/daylight_all_events_overlays/
#
#SBATCH --job-name=make_overlay
#SBATCH --array=1-32
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --account=kanarde_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --output=job_outputs/make_overlay.%A.%a.out
#SBATCH --error=job_outputs/make_overlay.%A.%a.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
export POSEIDON_THREADS="$NPROC"

IMAGE_DIR="$SITE_DATA/images/daylight_all_events"
FILE_LIST=$(file_list_for_task "$IMAGE_DIR/job_file_lists")
echo "Task $TASK_ID: running overlay_generator on $FILE_LIST"

SECONDS=0
"$BIN_DIR/overlay_generator" \
    "$IMAGE_DIR" \
    "$SITE_DATA/images/daylight_all_events_preds" \
    "$SITE_DATA/images/daylight_all_events_overlays" \
    "$FILE_LIST" \
    0.6  # overlay opacity
echo "Task $TASK_ID: $(wc -l < "$FILE_LIST") files in $SECONDS s"
