#! /bin/bash

#BSUB -J gen_flood_folders
#BSUB -W 15
#BSUB -n 1
#BSUB -q ccee
#BSUB -o gen_flood_folders.%J.out
#BSUB -e gen_flood_folders.%J.err

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

RUNNER_SCRIPT="$REPO_ROOT/poseidon_deploy/naiads/run_flood_event_setup.py"

LOCATION_FOLDER="$REPO_ROOT/data/down_east"
ABBR_EVENTS="$LOCATION_FOLDER/abbr_flood_events.csv"
FILT_ABBR_CSV="$LOCATION_FOLDER/abbr_flood_events.csv"
FLOOD_EVENTS_CSV="$LOCATION_FOLDER/flood_events.csv"

IMAGE_DIR="$LOCATION_FOLDER/images/daylight_all_events"
LABEL_DIR="$LOCATION_FOLDER/images/daylight_all_events_labels"
OUTPUT_DIR="$LOCATION_FOLDER/flood_events"

echo "Starting image organizer Python script..."
python -u $RUNNER_SCRIPT \
    --abbr_events_csv $ABBR_EVENTS \
    --filtered_abbr_csv $FILT_ABBR_CSV \
    --full_sensor_csv $FLOOD_EVENTS_CSV \
    --image_dir $IMAGE_DIR \
    --label_dir $LABEL_DIR \
    --output_dir $OUTPUT_DIR \
    --image_subfolder 'orig_images' \
    --label_subfolder 'labels' \
    --start_hour 6 \
    --end_hour 19 \
    --padding_hours 3

echo "Deactivating conda environment..."
conda deactivate

echo "Job finished."
