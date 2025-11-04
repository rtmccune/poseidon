#! /bin/bash

#BSUB -J image_pull
#BSUB -W 10
#BSUB -n 1
#BSUB -q ccee
#BSUB -o image_pull.%J.out
#BSUB -e image_pull.%J.err

echo "Activating conda environment..."
source ~/.bashrc
conda activate /rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/conda/poseidon

REPO_ROOT=$(cd $LS_SUBCWD/../.. && pwd)

RUNNER_SCRIPT="$REPO_ROOT/poseidon_deploy/naiads/run_image_pull.py"
EVENT_CSV="$REPO_ROOT/data/carolina_beach/abbr_flood_events.csv"

IMAGE_DRIVE='/rsstu/users/k/kanarde/Sunnyverse-Images'
OUTPUT_DIR="$REPO_ROOT/data/carolina_beach/images/all_events_during_FOV"

echo "Starting photo pull Python script..."
python $RUNNER_SCRIPT \
    --drive $IMAGE_DRIVE \
    --dest $OUTPUT_DIR \
    --csv $EVENT_CSV \
    --buffer 1.5

echo "Deactivating conda environment..."
conda deactivate

echo "Job finished."
