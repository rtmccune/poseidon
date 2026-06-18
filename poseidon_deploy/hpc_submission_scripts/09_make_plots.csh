#! /bin/bash

#BSUB -J plot_batch[1-19] ####--- EDIT THIS to match your number of event folders ---####
#BSUB -W 30
#BSUB -n 4
#BSUB -R "rusage[mem=12G]"
#BSUB -q ccee
#BSUB -o job_outputs/plotting.%J.%I.out
#BSUB -e job_outputs/plotting.%J.%I.err

source ~/.bashrc

export MPI4PY_RC_INITIALIZE=False
export TMPDIR=/tmp
export PROJ_NETWORK=OFF

SUBMIT_DIR="${LS_SUBCWD:-$PWD}"
mkdir -p $SUBMIT_DIR/job_outputs

ENV_FILE="$SUBMIT_DIR/../hpc_paths.env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo "Loaded environment variables from $ENV_FILE"
else
    echo "Warning: No hpc_paths.env file found"
fi

echo "Activating conda environment..."
conda activate $POSEIDON_ENV

REPO_ROOT=$(cd $LS_SUBCWD/../.. && pwd)
RUNNER_SCRIPT="$REPO_ROOT/poseidon_deploy/naiads/run_plotter_single_event.py"
EVENT_DIR="$REPO_ROOT/data/carolina_beach/flood_events"
BASEMAP_FILE="/share/kanarde/rmccune/poseidon/data/basemaps/CB_03_basemap.tif"

# --- ARRAY LOGIC ---
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

echo "Starting plotter Python script..."
python -u $RUNNER_SCRIPT \
    --target_event_dir $TARGET_EVENT_DIR \
    --location "CB_03" \
    --basemap $BASEMAP_FILE \
    --min_x 712160 \
    --max_x 712230 \
    --min_y 33100 \
    --max_y 33170 \
    --bbox_crs "EPSG:32119" \
    --resolution 0.05 \
    --stats "95_perc"

echo "Deactivating conda environment..."
conda deactivate
echo "Job finished."
