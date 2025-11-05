#! /bin/bash

#BSUB -J image_filter
#BSUB -W 15
#BSUB -n 1
#BSUB -q ccee
#BSUB -o image_filter.%J.out
#BSUB -e image_filter.%J.err

source ~/.bashrc
ENV_FILE="${LS_SUBCWD:-$PWD}/hpc_paths.env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
else
    echo "Warning: No .env file found at $ENV_FILE"
fi

echo "Activating conda environment..."
conda activate $POSEIDON_ENV

REPO_ROOT=$(cd $LS_SUBCWD/../.. && pwd)

RUNNER_SCRIPT="$REPO_ROOT/poseidon_deploy/naiads/run_image_filter.py"

IMAGE_DRIVE='/rsstu/users/k/kanarde/Sunnyverse-Images'
IMAGE_DIR="$REPO_ROOT/data/down_east/images/all_events_during_FOV"
OUTPUT_DIR="$REPO_ROOT/data/down_east/images/daylight_all_events"

echo "Starting photo filter Python script..."
python $RUNNER_SCRIPT \
    --drive $IMAGE_DRIVE \
    --image_dir $IMAGE_DIR \
    --dest $OUTPUT_DIR \
    --start 6 \
    --end 19

echo "Deactivating conda environment..."
conda deactivate

echo "Job finished."
