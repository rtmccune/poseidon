#! /bin/bash

#BSUB -J depths_batch[1-19] ####--- EDIT THIS to match your number of event folders ---####
#BSUB -W 45
#BSUB -n 2
#BSUB -R "rusage[mem=8G]"   ####--- 16G is safer for CuPy/NumPy array operations ---####
#BSUB -R "select[a100 || l40 || l40s || h100]"
#BSUB -gpu "num=1:mode=shared"
#BSUB -q gpu
#BSUB -o job_outputs/depths.%J.%I.out
#BSUB -e job_outputs/depths.%J.%I.err

source ~/.bashrc

module load cuda/11.2
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

# Point to your newly updated single-event Python script
RUNNER_SCRIPT="$REPO_ROOT/poseidon_deploy/naiads/run_calc_depths_single_event.py"

GRID_DIR="$REPO_ROOT/data/grids"
EVENT_DIR="$REPO_ROOT/data/carolina_beach/flood_events"

# --- ARRAY LOGIC: Map Job Index to Event Folder ---
EVENTS=($(find "$EVENT_DIR" -mindepth 1 -maxdepth 1 -type d | sort))

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

echo "Starting Depth Calculation Python script..."
python -u $RUNNER_SCRIPT \
    --target_event_dir $TARGET_EVENT_DIR \
    --grid_dir $GRID_DIR \
    --grid_descr "carolina_beach" \
    --zarr_base "zarr" \
    --zarr_label_dir "labels_rects" \
    --zarr_depth_dir "depth_maps" \
    --plot_base_dir "plots"

echo "Deactivating conda environment..."
conda deactivate

echo "Job finished."
