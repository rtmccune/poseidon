#! /bin/bash

#BSUB -J image_filter
#BSUB -W 5
#BSUB -n 1
#BSUB -q ccee
#BSUB -o image_filter.%J.out
#BSUB -e image_filter.%J.err

echo "Activating conda environment..."
source ~/.bashrc
conda activate /rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/conda/poseidon

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
