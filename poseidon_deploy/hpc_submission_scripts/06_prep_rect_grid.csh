#! /bin/bash

#BSUB -J prep_grid
#BSUB -W 15
#BSUB -n 2
#BSUB -R "rusage[mem=4G]"
#BSUB -q ccee 
#BSUB -o prep_grid.%J.out
#BSUB -e prep_grid.%J.err

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

REPO_ROOT=$(cd $LS_SUBCWD/../.. && pwd)
PREP_SCRIPT="$REPO_ROOT/poseidon_deploy/naiads/run_prep_grid.py"

LIDAR_FILE="$REPO_ROOT/data/lidar/Job1051007_34077_04_88.laz"
GRID_DIR="$REPO_ROOT/data/grids"

echo "Starting grid prep..."
python -u $PREP_SCRIPT \
    --lidar_file $LIDAR_FILE \
    --min_x 712160 \
    --max_x 712230 \
    --min_y 33100 \
    --max_y 33170 \
    --grid_dir $GRID_DIR \
    --resolution 0.05 \
    --lidar_units "feet" \
    --grid_descr "carolina_beach"

echo "Job finished."
