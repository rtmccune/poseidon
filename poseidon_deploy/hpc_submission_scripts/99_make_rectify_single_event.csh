#! /bin/bash

#BSUB -J rectify_single
#BSUB -W 15
#BSUB -n 4
#BSUB -R "rusage[mem=4G]"
#BSUB -R "select[a100 || l40 || l40s || h100]"
#BSUB -gpu "num=1:mode=shared"
#BSUB -q gpu
#BSUB -o job_outputs/rectify_single.%J.out
#BSUB -e job_outputs/rectify_single.%J.err

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

# ==============================================================================
# PASTE YOUR MISSING FOLDER NAME HERE
# Example: MISSING_EVENT_NAME="DE_01_20240904110947_20240904192647"
# ==============================================================================
MISSING_EVENT_NAME="DE_01_20240821123141_20240821191541"


# Construct the full paths
RUNNER_SCRIPT="$REPO_ROOT/poseidon_deploy/naiads/run_rectify_single_event.py"
LIDAR_FILE="$REPO_ROOT/data/lidar/down_east.laz"
GRID_DIR="$REPO_ROOT/data/grids"
TARGET_EVENT_DIR="$REPO_ROOT/data/down_east/flood_events/$MISSING_EVENT_NAME"

echo "=================================================="
echo "Processing Single Event Directory: ${TARGET_EVENT_DIR}"
echo "=================================================="

# Quick check to make sure you didn't accidentally leave the placeholder or make a typo
if [ ! -d "${TARGET_EVENT_DIR}" ]; then
    echo "ERROR: Target directory not found: ${TARGET_EVENT_DIR}"
    echo "Did you update the MISSING_EVENT_NAME variable?"
    exit 1
fi

echo "Starting image rectifier Python script..."
python -u $RUNNER_SCRIPT \
    --lidar_file $LIDAR_FILE \
    --target_event_dir $TARGET_EVENT_DIR \
    --min_x 847809.694 \
    --max_x 847973.874 \
    --min_y 127254.634 \
    --max_y 127450.141 \
    --camera_name "DE_01" \
    --intrinsics_name "suds_cam" \
    --grid_dir $GRID_DIR \
    --resolution 0.05 \
    --lidar_units "meters" \
    --grid_descr "down_east" \
    --image_subfolder 'orig_images' \
    --label_subfolder 'labels' \
    --zarr_base "zarr" \
    --zarr_orig_name "orig_image_rects" \
    --zarr_label_name "labels_rects"

echo "Deactivating conda environment..."
conda deactivate

echo "Job finished."
