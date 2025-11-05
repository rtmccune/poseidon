#!/bin/bash

#BSUB -J "make_labels[1-16]"  # <-- EDIT to the exact number of file lists you created.
#BSUB -W 60
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -q ccee
#BSUB -o job_outputs/make_labels.%J.%I.out
#BSUB -e job_outputs/make_labels.%J.%I.err

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
PREDS_DIR="$REPO_ROOT/data/carolina_beach/images/daylight_all_events_preds"
LISTS_DIR="$PREDS_DIR/job_file_lists"

OUTPUT_DIR="$REPO_ROOT/data/carolina_beach/images/daylight_all_events_labels"


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
    "${PREDS_DIR}" \
    "${FILE_LIST}" \
    "${OUTPUT_DIR}" \

echo "------------------------------------------------"
echo "Execution finished for Job Index ${LSB_JOBINDEX}."
echo "Job Ended: $(date)"
echo "------------------------------------------------"
