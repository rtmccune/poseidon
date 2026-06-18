#! /bin/bash

#BSUB -J stitch_comparisons
#BSUB -W 60                          # Bumped to 60 minutes for image I/O safety
#BSUB -n 8                           # Increased to 8 cores for parallel processing
#BSUB -R "span[hosts=1]"             # CRITICAL: Ensures all 8 cores are on the same node
#BSUB -R "rusage[mem=8G]"            # Increased memory for OpenCV image handling
#BSUB -q ccee 
#BSUB -o stitch_images.%J.out
#BSUB -e stitch_images.%J.err

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

# Assuming you place the new Python script in your utils folder
STITCH_SCRIPT="$REPO_ROOT/poseidon_utils/stitch_comparisons.py"

# UPDATE THESE to point to the actual folders holding your images
ORIG_DIR="$REPO_ROOT/data/originals"
OVERLAY_DIR="$REPO_ROOT/data/overlays"
OUTPUT_DIR="$REPO_ROOT/data/stitched_comparisons"

echo "Starting parallel image stitching..."

# $LSB_DJOB_NUMPROC automatically grabs the '-n 8' value defined above
python -u $STITCH_SCRIPT \
    "$ORIG_DIR" \
    "$OVERLAY_DIR" \
    "$OUTPUT_DIR" \
    --workers $LSB_DJOB_NUMPROC

echo "Job finished."