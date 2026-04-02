#! /bin/bash

#BSUB -J plot_depths
#BSUB -W 240
#BSUB -n 64
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24G]"
#BSUB -gpu "num=1:mode=shared"
#BSUB -q gpu    
#BSUB -o plotting.%J.out
#BSUB -e plotting.%J.err

source ~/.bashrc

export MPI4PY_RC_INITIALIZE=False
export TMPDIR=/tmp
export PROJ_NETWORK=OFF

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
BASEMAP_FILE="/share/jcdietri/rmccune/poseidon/data/basemaps/DE_01_basemap.tif"

echo "Starting plotter Python script with MPI..."

# Using CB_03 extents based on your snippet
# Change --location to "DE_01" and update extents if running for Down East
mpirun python -u $RUNNER_SCRIPT \
    --event_dir $EVENT_DIR \
    --location "DE_01" \
    --basemap $BASEMAP_FILE \
    --min_x 847809.694 \
    --max_x 847973.874 \
    --min_y 127254.634 \
    --max_y 127450.141 \
    --bbox_crs "EPSG:32119" \
    --resolution 0.05 \
    --stats "95_perc"

echo "Deactivating conda environment..."
conda deactivate

echo "Job finished."
