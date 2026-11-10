#!/bin/bash
# Stage 03: convert segmentation predictions to grayscale label images.
#
# Runs the compiled pred_label_generator on one file list per array
# task. Create lists of the prediction folder first, e.g.
#   LIST_SOURCE=$SITE_DATA/images/daylight_all_events_preds \
#       bash 99_create_job_lists.sh
# then submit with `sbatch 03_make_labels.sh`; --array must match the
# number of lists (NUM_JOBS, default 32).
#
# Each image is a colour lookup and a PNG write, so a list of ~160
# images takes a few minutes on 8 CPUs. The tool starts one thread per
# CPU given to the task (POSEIDON_THREADS); rebuild it (README: C++
# Executables) if its log says it is using every core on the node.
#
# Input:  $SITE_DATA/images/daylight_all_events_preds/
# Output: $SITE_DATA/images/daylight_all_events_labels/
#
#SBATCH --job-name=make_labels
#SBATCH --array=1-32
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --account=kanarde_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --output=job_outputs/make_labels.%A.%a.out
#SBATCH --error=job_outputs/make_labels.%A.%a.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
export POSEIDON_THREADS="$NPROC"

PREDS_DIR="$SITE_DATA/images/daylight_all_events_preds"
FILE_LIST=$(file_list_for_task "$PREDS_DIR/job_file_lists")
echo "Task $TASK_ID: running pred_label_generator on $FILE_LIST"

SECONDS=0
"$BIN_DIR/pred_label_generator" \
    "$PREDS_DIR" \
    "$FILE_LIST" \
    "$SITE_DATA/images/daylight_all_events_labels"
echo "Task $TASK_ID: $(wc -l < "$FILE_LIST") files in $SECONDS s"
