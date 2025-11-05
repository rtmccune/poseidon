#! /bin/bash

#BSUB -J file_org
#BSUB -W 30
#BSUB -n 1
#BSUB -q ccee
#BSUB -o file_org.%J.out
#BSUB -e file_org.%J.err

echo "Activating conda environment..."
source ~/.bashrc
conda activate /rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/conda/poseidon

REPO_ROOT=$(cd $LS_SUBCWD/../.. && pwd)

RUNNER_SCRIPT="$REPO_ROOT/poseidon_deploy/naiads/run_flood_event_setup.py"

ABBR_EVENTS="$REPO_ROOT/data/down_east/abbr_flood_events.csv"
FILT_ABBR_CSV="$REPO_ROOT/data/down_east/filt_abbr_flood_events.csv"
FLOOD_EVENTS_CSV="$REPO_ROOT/data/down_east/flood_events.csv"

IMAGE_DIR="$REPO_ROOT/data/down_east/images/all_events_during_FOV"
OUTPUT_DIR="$REPO_ROOT/data/down_east/flood_events"

echo "Starting image organizer Python script..."
python $RUNNER_SCRIPT \
    --abbr_events_csv $ABBR_EVENTS \
    --filtered_abbr_csv $FILT_ABBR_CSV \
    --full_sensor_csv $FLOOD_EVENTS_CSV \
    --image_dir $IMAGE_DIR \
    --dest $OUTPUT_DIR \
    --image_subfolder 'orig_images' \
    --start_hour 6 \
    --end_hour 19 \
    --padding_hours 3

echo "Deactivating conda environment..."
conda deactivate

echo "Job finished."
