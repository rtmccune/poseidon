#! /bin/bash

#BSUB -J roadway_access
#BSUB -W 15
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4G]"
#BSUB -q ccee
#BSUB -o summary_stats_access.%J.out
#BSUB -e summary_stats_access.%J.err

source ~/.bashrc

# --- SETUP ---
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

# --- RUN ---
# Assumes you saved the python code as 'summarize_roadway_stats.py' 
# in the same directory as this script.
SCRIPT_NAME="summary_road_access_stats.py"

echo "Starting Summary Statistics..."
python -u $SCRIPT_NAME

echo "Done."
