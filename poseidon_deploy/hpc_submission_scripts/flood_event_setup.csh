#! /bin/bash

#BSUB -J gen_flood_folders
#BSUB -W 60
#BSUB -n 1
#BSUB -q ccee
#BSUB -o gen_flood_folders.%J.out
#BSUB -e gen_flood_folders.%J.err

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

RUNNER_SCRIPT="$REPO_ROOT/poseidon_deploy/naiads/run_flood_event_setup.py"

ABBR_EVENTS="$REPO_ROOT/data/carolina_beach/abbr_flood_events.csv"
FILT_ABBR_CSV="$REPO_ROOT/data/carolina_beach/filt_abbr_flood_events.csv"
FLOOD_EVENTS_CSV="$REPO_ROOT/data/carolina_beach/flood_events.csv"

IMAGE_DIR="$REPO_ROOT/data/carolina_beach/images/daylight_all_events"
OUTPUT_DIR="$REPO_ROOT/data/carolina_beach/flood_events"

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
