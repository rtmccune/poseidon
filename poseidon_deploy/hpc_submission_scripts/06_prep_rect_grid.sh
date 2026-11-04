#!/bin/bash
# Stage 06: build the site's elevation grid from LiDAR (run once).
#
# Input:  the site config's lidar_file and grid extent
# Output: $REPO_ROOT/data/grids/shared_grid_$SITE.npz
#
#BSUB -J prep_grid
#BSUB -W 15
#BSUB -n 2
#BSUB -R "rusage[mem=4G]"
#BSUB -q ccee
#BSUB -o prep_grid.%J.out
#BSUB -e prep_grid.%J.err

source "${LS_SUBCWD:-$PWD}/_common.sh"
require_site
activate_env

python -u "$NAIADS/run_prep_grid.py" \
    --site "$SITE" \
    --grid_dir "$REPO_ROOT/data/grids"
