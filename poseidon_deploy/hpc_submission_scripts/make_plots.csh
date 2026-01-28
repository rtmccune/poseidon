#! /bin/bash

#BSUB -J plot_depths
#BSUB -W 120
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8G]"
#BSUB -R "select[a100 || l40 || l40s || h100]"
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
EVENT_DIR="$REPO_ROOT/data/carolina_beach/flood_events"
<<<<<<< HEAD
=======
BASEMAP_FILE="/share/jcdietri/rmccune/poseidon/data/basemaps/CB_03_basemap.tif"
>>>>>>> 1f515a37b5e3bc753d20192416bfb2e646210042

echo "Starting plotter Python script with MPI..."

# Using CB_03 extents based on your snippet
# Change --location to "DE_01" and update extents if running for Down East
mpirun python -u $RUNNER_SCRIPT \
    --event_dir $EVENT_DIR \
    --location "CB_03" \
    --basemap $BASEMAP_FILE \
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
