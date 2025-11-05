#!/bin/bash

#BSUB -J "make_overlay[1-16]"  # <-- EDIT to the exact number of file lists you created.
#BSUB -W 10
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8]"
#BSUB -q ccee
#BSUB -o job_outputs/make_overlay.%J.%I.out
#BSUB -e job_outputs/make_overlay.%J.%I.err

source ~/.bashrc

# Exit immediately if a command exits with a non-zero status.
set -e

echo "------------------------------------------------"
echo "Job Started: $(date)"
echo "Job ID: ${LSB_JOBID}"
echo "Job Index: ${LSB_JOBINDEX}"
echo "Running on host: $(hostname)"
echo "------------------------------------------------"

# Path to your compiled C++ executable
REPO_ROOT=$(cd $LS_SUBCWD/../.. && pwd)
EXEC_PATH="$REPO_ROOT/poseidon_utils/bin/overlay_generator"

# Base directories for your data
IMAGE_DIR="$REPO_ROOT/data/carolina_beach/images/daylight_all_events"
LISTS_DIR="$IMAGE_DIR/job_file_lists"

PREDS_DIR="$REPO_ROOT/data/carolina_beach/images/daylight_all_events_preds"
OUTPUT_DIR="$REPO_ROOT/data/carolina_beach/images/daylight_all_events_overlays"


# Use the LSF-provided job index to construct the path to the correct file list.
# Task 1 will get file_list_1.txt, Task 2 gets file_list_2.txt, and so on.
FILE_LIST="${LISTS_DIR}/file_list_${LSB_JOBINDEX}.txt"

echo "This task will process the file list: ${FILE_LIST}"

# Sanity check that file list exists
if [ ! -f "${FILE_LIST}" ]; then
    echo "FATAL ERROR: File list not found: ${FILE_LIST}"
    exit 1
fi

echo "Executing overlay generator..."

"${EXEC_PATH}" \
    "${IMAGE_DIR}" \
    "${PREDS_DIR}" \
    "${OUTPUT_DIR}" \
    "${FILE_LIST}" \
    0.6 # Optional alpha value

echo "------------------------------------------------"
echo "Execution finished for Job Index ${LSB_JOBINDEX}."
echo "Job Ended: $(date)"
echo "------------------------------------------------"
