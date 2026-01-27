#! /bin/bash

#BSUB -J plot_depths
#BSUB -W 120
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8G]"
#BSUB -q ccee
#BSUB -o plotting.%J.out
#BSUB -e plotting.%J.err

source ~/.bashrc

export MPI4PY_RC_INITIALIZE=False

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
RUNNER_SCRIPT="$REPO_ROOT/poseidon_deploy/naiads/run_plotter.py"
EVENT_DIR="$REPO_ROOT/data/down_east/flood_events"

echo "Starting plotter Python script with MPI..."

# Using CB_03 extents based on your snippet
# Change --location to "DE_01" and update extents if running for Down East
mpirun python -u $RUNNER_SCRIPT \
    --event_dir $EVENT_DIR \
    --location "CB_03" \
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
