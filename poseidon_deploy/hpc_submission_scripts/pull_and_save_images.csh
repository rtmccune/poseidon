#! /bin/bash

#BSUB -J image_pull
#BSUB -W 10
#BSUB -n 1
#BSUB -q ccee
#BSUB -o image_pull.%J.out
#BSUB -e image_pull.%J.err

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

REPO_ROOT=$(cd $LS_SUBCWD/../.. && pwd)

RUNNER_SCRIPT="$REPO_ROOT/poseidon_deploy/naiads/run_image_pull.py"
EVENT_CSV="$REPO_ROOT/data/carolina_beach/abbr_flood_events.csv"

OUTPUT_DIR="$REPO_ROOT/data/carolina_beach/images/all_events_during_FOV"

echo "Starting photo pull Python script..."
python $RUNNER_SCRIPT \
    --drive $IMAGE_DRIVE \
    --dest $OUTPUT_DIR \
    --csv $EVENT_CSV \
    --buffer 3

echo "Deactivating conda environment..."
conda deactivate

echo "Job finished."
