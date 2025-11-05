#! /bin/bash

#BSUB -J make_jobs
#BSUB -W 15
#BSUB -n 1
#BSUB -q ccee
#BSUB -o make_jobs.%J.out
#BSUB -e make_jobs.%J.err

source ~/.bashrc

# Resolve the directory where the job was submitted from (LSF variable or fallback)
SUBMIT_DIR="${LS_SUBCWD:-$PWD}"

# Point one directory up from that
ENV_FILE="$SUBMIT_DIR/../hpc_paths.env"

# Load the env file if it exists
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo "Loaded environment variables from $ENV_FILE"
else
    echo "Warning: No hpc_paths.env file found at $ENV_FILE"
fi

echo "Activating conda environment..."
conda activate $POSEIDON_ENV

REPO_ROOT=$(cd $LS_SUBCWD/../.. && pwd)

RUNNER_SCRIPT="$REPO_ROOT/poseidon_deploy/naiads/run_create_file_lists.py"

IMAGE_DIR="$REPO_ROOT/data/carolina_beach/images/daylight_all_events"
OUTPUT_DIR="$REPO_ROOT/data/carolina_beach/images/daylight_all_events/job_file_lists"

echo "Starting photo filter Python script..."
python -u $RUNNER_SCRIPT \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --num_jobs 16 \

echo "Deactivating conda environment..."
conda deactivate

echo "Job finished."
