#! /bin/bash

#BSUB -J plot_access
#BSUB -W 15                            # 15 minutes is plenty for generating these plots
#BSUB -n 1                             # Single core (script is not parallelized)
#BSUB -R "rusage[mem=4G]"              # 4GB memory is safe for pandas CSV loading and matplotlib
#BSUB -q ccee 
#BSUB -o job_outputs/poseidon_plots.%J.out
#BSUB -e job_outputs/poseidon_plots.%J.err

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

# Pointing to where you save the new Python script
PLOT_SCRIPT="$REPO_ROOT/poseidon_utils/access_plotter.py"

# Update this path to point to the parent directory that CONTAINS the 'flood_events' folder
# (Assuming it sits inside your carolina_beach data directory)
TARGET_DIR="$REPO_ROOT/data/down_east/"

echo "Navigating to data directory..."
cd "$TARGET_DIR" || { echo "Failed to navigate to $TARGET_DIR"; exit 1; }

echo "Starting POSEIDON visual generation..."

# Run the plotting script
python -u "$PLOT_SCRIPT"

echo "Job finished. Check the 'plots' folder inside $TARGET_DIR."
