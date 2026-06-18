#! /bin/bash

#BSUB -J rectify_batch[1-19] ####--- EDIT THIS: Match the total number of event folders ---####
#BSUB -W 10                  ####--- 1 hour should be plenty for a single folder ---####
#BSUB -n 4                   
#BSUB -R "rusage[mem=4G]"
#BSUB -R "select[a100 || l40 || l40s || h100]"
#BSUB -gpu "num=1:mode=shared"
#BSUB -q gpu
#BSUB -o job_outputs/rectify.%J.%I.out
#BSUB -e job_outputs/rectify.%J.%I.err

source ~/.bashrc

module load cuda/12.6
export MPI4PY_RC_INITIALIZE=False

# Resolve directories
SUBMIT_DIR="${LS_SUBCWD:-$PWD}"
mkdir -p $SUBMIT_DIR/job_outputs

ENV_FILE="$SUBMIT_DIR/../hpc_paths.env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo "Loaded environment variables from $ENV_FILE"
else
    echo "Warning: No hpc_paths.env file found at $ENV_FILE"
fi

echo "Activating conda environment..."
conda activate $POSEIDON_ENV

REPO_ROOT=$(cd $LS_SUBCWD/../.. && pwd)

# Point to your newly updated Python script
RUNNER_SCRIPT="$REPO_ROOT/poseidon_deploy/naiads/run_rectify_single_event.py"

LIDAR_FILE="$REPO_ROOT/data/lidar/Job1051007_34077_04_88.laz"
GRID_DIR="$REPO_ROOT/data/grids"
EVENT_DIR="$REPO_ROOT/data/carolina_beach/flood_events"

# --- ARRAY LOGIC: Map Job Index to Event Folder ---
# Get an array of all subdirectories inside EVENT_DIR, sorted alphabetically
EVENTS=($(find "$EVENT_DIR" -mindepth 1 -maxdepth 1 -type d | sort))

# Bash arrays are 0-indexed, but LSB_JOBINDEX starts at 1
ARRAY_INDEX=$((LSB_JOBINDEX - 1))
TARGET_EVENT_DIR=${EVENTS[$ARRAY_INDEX]}

echo "=================================================="
echo "Job Index: ${LSB_JOBINDEX}"
echo "Processing Event Directory: ${TARGET_EVENT_DIR}"
echo "=================================================="

if [ ! -d "${TARGET_EVENT_DIR}" ]; then
    echo "ERROR: Target directory not found: ${TARGET_EVENT_DIR}"
    exit 1
fi

echo "Starting image rectifier Python script..."
python -u $RUNNER_SCRIPT \
    --lidar_file $LIDAR_FILE \
    --target_event_dir $TARGET_EVENT_DIR \
    --min_x 712160 \
    --max_x 712230 \
    --min_y 33100 \
    --max_y 33170 \
    --camera_name "CB_03" \
    --intrinsics_name "suds_cam" \
    --grid_dir $GRID_DIR \
    --resolution 0.05 \
    --lidar_units "feet" \
    --grid_descr "carolina_beach" \
    --image_subfolder 'orig_images' \
    --label_subfolder 'labels' \
    --zarr_base "zarr" \
    --zarr_orig_name "orig_image_rects" \
    --zarr_label_name "labels_rects"

echo "Deactivating conda environment..."
conda deactivate

echo "Job finished."
