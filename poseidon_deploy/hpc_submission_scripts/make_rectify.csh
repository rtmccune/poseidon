#! /bin/bash

#BSUB -J rectify
#BSUB -W 30
#BSUB -n 12
#BSUB -R "rusage[mem=4G]"
#BSUB -gpu "num=1:mode=shared"
#BSUB -q gpu
#BSUB -o rectification.%J.out
#BSUB -e rectification.%J.err

source ~/.bashrc

module load cuda/12.6
export MPI4PY_RC_INITIALIZE=False

# Resolve the directory where the job was submitted from (LSF variable or fallback)
SUBMIT_DIR="${LS_SUBCWD:-$PWD}"

# Point one directory up from that
ENV_FILE="$SUBMIT_DIR/../hpc_paths.env"

# Load the env file if it exists
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo "Loaded environment variables from $ENV_FILE"
else
    echo "Warning: No hpc_paths.env file found at $ENV_FILE"
fi

echo "Activating conda environment..."
conda activate $POSEIDON_ENV

REPO_ROOT=$(cd $LS_SUBCWD/../.. && pwd)

RUNNER_SCRIPT="$REPO_ROOT/poseidon_deploy/naiads/run_rectify.py"

LIDAR_FILE="$REPO_ROOT/data/lidar/combined_point_cloud_down_east.laz"
GRID_DIR="$REPO_ROOT/data/grids"
EVENT_DIR="$REPO_ROOT/data/down_east/flood_events"

echo "Starting image rectifier Python script..."
python -u $RUNNER_SCRIPT \
    --lidar_file $LIDAR_FILE \
    --event_dir $EVENT_DIR \
    --min_x 847809.694 \
    --max_x 847973.874 \
    --min_y 127254.634 \
    --max_y 127450.141 \
    --camera_name "DE_01" \
    --intrinsics_name "suds_cam" \
    --grid_dir $GRID_DIR \
    --resolution 0.05 \
    --lidar_units "meters" \
    --grid_descr "down_east" \
    --image_subfolder 'orig_images' \
    --label_subfolder 'labels' \
    --zarr_base "zarr" \
    --zarr_orig_name "orig_image_rects" \
    --zarr_label_name "labels_rects"

echo "Deactivating conda environment..."
conda deactivate

echo "Job finished."
