#! /bin/bash

#BSUB -J image_pull
#BSUB -W 10
#BSUB -n 1
#BSUB -q ccee
#BSUB -o image_pull.%J.out
#BSUB -e image_pull.%J.err

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
