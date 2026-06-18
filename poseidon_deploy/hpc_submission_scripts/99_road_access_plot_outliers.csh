#! /bin/bash

#BSUB -J anomalies
#BSUB -W 15                            # 15 minutes is plenty for statistical calculations
#BSUB -n 1                             # Single core (script is not parallelized)
#BSUB -R "rusage[mem=4G]"              # 4GB memory is safe for pandas CSV loading and numpy
#BSUB -q ccee 
#BSUB -o job_outputs/anomalies.%J.out
#BSUB -e job_outputs/anomalies.%J.err

source ~/.bashrc

SUBMIT_DIR="${LS_SUBCWD:-$PWD}"
mkdir -p $SUBMIT_DIR/job_outputs

ENV_FILE="$SUBMIT_DIR/../hpc_paths.env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
else
    echo "Warning: No hpc_paths.env file found"
fi

echo "Activating conda environment..."
conda activate $POSEIDON_ENV

# Define paths relative to the repository root
REPO_ROOT=$(cd $LS_SUBCWD/../.. && pwd)

# Pointing to where you saved the new Python script
PLOT_SCRIPT="$REPO_ROOT/poseidon_utils/find_stat_anomalies.py"

# Target directory containing the 'flood_events' folder
TARGET_DIR="$REPO_ROOT/data/down_east/"

echo "Navigating to data directory..."
cd "$TARGET_DIR" || { echo "Failed to navigate to $TARGET_DIR"; exit 1; }

echo "Starting POSEIDON anomaly detection..."

# Run the anomaly hunting script
python -u "$PLOT_SCRIPT"

echo "Job finished. Check $TARGET_DIR for the anomaly report and diagnostic plot."
