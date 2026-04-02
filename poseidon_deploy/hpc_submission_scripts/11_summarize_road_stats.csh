#! /bin/bash

#BSUB -J stats
#BSUB -n 1
#BSUB -W 15
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4G]"
#BSUB -q ccee
#BSUB -o summary_stats.%J.out
#BSUB -e summary_stats.%J.err

source ~/.bashrc

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

# --- CONFIGURATION ---
REPO_ROOT=$(cd $LS_SUBCWD/../.. && pwd)
SCRIPT_NAME="$REPO_ROOT/poseidon_deploy/naiads/run_road_access_stats.py"

# Comment/Uncomment to swap locations
# EVENT_DIR="$REPO_ROOT/data/carolina_beach/flood_events"
EVENT_DIR="$REPO_ROOT/data/down_east/flood_events"

# The name of the transect (matches the JSON file name without '.json')
TRANSECT_NAME="shell_hill_rd_polygon"

# --- RUN ---
echo "Starting Summary Statistics..."
python -u $SCRIPT_NAME \
    --event_dir $EVENT_DIR \
    --transect_name $TRANSECT_NAME

echo "Done."
