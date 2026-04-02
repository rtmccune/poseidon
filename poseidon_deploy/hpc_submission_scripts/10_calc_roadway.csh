#! /bin/bash

#BSUB -J roadway_batch[1-19] ####--- EDIT THIS to match your number of event folders ---####
#BSUB -W 45
#BSUB -n 4
#BSUB -R "rusage[mem=16G]"
#BSUB -q ccee  ####--- Standard CPU queue ---####
#BSUB -o job_outputs/roadway.%J.%I.out
#BSUB -e job_outputs/roadway.%J.%I.err

source ~/.bashrc

export TMPDIR=/tmp

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
RUNNER_SCRIPT="$REPO_ROOT/poseidon_deploy/naiads/run_roadway_analyzer_single_event.py"

EVENT_DIR="$REPO_ROOT/data/carolina_beach/flood_events"
JSON_FILE="$REPO_ROOT/data/transects/canal_dr_polygon.json"

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

echo "Starting Roadway Analysis..."
python -u $RUNNER_SCRIPT \
    --target_event_dir $TARGET_EVENT_DIR \
    --json_path $JSON_FILE \
    --label "roadway" \
    --step_size 1.0 \
    --statistic "95_perc"

echo "Deactivating conda environment..."
conda deactivate
echo "Job Finished"
