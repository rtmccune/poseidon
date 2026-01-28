#! /bin/bash

#BSUB -J roadway_calc
#BSUB -W 120
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8G]"
#BSUB -q ccee
#BSUB -o roadway.%J.out
#BSUB -e roadway.%J.err

source ~/.bashrc

export MPI4PY_RC_INITIALIZE=False
export TMPDIR=/tmp

SUBMIT_DIR="${LS_SUBCWD:-$PWD}"
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
RUNNER_SCRIPT="$REPO_ROOT/poseidon_deploy/naiads/run_roadway_analyzer.py"

# Point to data on /rsstu
EVENT_DIR="/rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/conda/poseidon/data/down_east/flood_events"

# Define your LabelMe JSON file location
# (Upload this file to /share or /rsstu before running)
JSON_FILE="$REPO_ROOT/data/transects/roadway_transect_CB03.json"

echo "Activating conda environment..."
conda activate $POSEIDON_ENV

echo "Starting Roadway Analysis..."
mpirun python -u $RUNNER_SCRIPT \
    --event_dir $EVENT_DIR \
    --json_path $JSON_FILE \
    --label "roadway" \
    --step_size 1.0

echo "Job Finished"