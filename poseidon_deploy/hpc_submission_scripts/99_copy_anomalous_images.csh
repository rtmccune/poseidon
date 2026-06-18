#! /bin/bash

#BSUB -J copy_imgs
#BSUB -W 15                            # 15 minutes is plenty for copying images
#BSUB -n 1                             # Single core (script is not parallelized)
#BSUB -R "rusage[mem=4G]"              # 4GB memory is safe for pandas CSV loading and file I/O
#BSUB -q ccee 
#BSUB -o job_outputs/copy_imgs.%J.out
#BSUB -e job_outputs/copy_imgs.%J.err

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

# Pointing to where you saved the new copying script
COPY_SCRIPT="$REPO_ROOT/poseidon_utils/copy_anomalous_images.py"

# Target directory containing both the anomaly reports and the 'images' folder
TARGET_DIR="$REPO_ROOT/data/carolina_beach/"

echo "Navigating to data directory..."
cd "$TARGET_DIR" || { echo "Failed to navigate to $TARGET_DIR"; exit 1; }

echo "Starting anomalous image extraction..."

# Run the image copying script
python -u "$COPY_SCRIPT"

echo "Job finished. Check $TARGET_DIR/images/anomaly_comparisons/ for the isolated images."
