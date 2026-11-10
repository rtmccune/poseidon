#!/bin/bash
# Analysis: site description figure (North Carolina, communities, sites).
#
# Compute nodes have no internet, so download the background layers
# first where there is (e.g. your own machine), from the repo root:
#   python poseidon_deploy/analysis/site_description_figure.py fetch \
#       --sites carolina_beach down_east --layers_dir data/site_figure/layers
# then copy data/site_figure/layers/ to the same place on the HPC and
# submit with `sbatch 99_site_description_figure.sh`. Rendering needs
# only local files: the layers, data/grids/shared_grid_<site>.npz
# (stage 06) and a camera photo per site (data/<site>/CAM_*.jpg).
# Options after the script name go to the render step, e.g.
#   sbatch 99_site_description_figure.sh --elevation_label "Elevation (m NAVD88)"
#
# Ignores SITE in hpc_paths.env; set LEFT_SITE / RIGHT_SITE to change
# the sites (default carolina_beach on the left, down_east on the right).
# Output: data/site_figure/site_description.{png,pdf,svg} and
#         attribution.txt (imagery credits for the caption)
#
#SBATCH --job-name=site_figure
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=kanarde_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --output=job_outputs/site_figure.%j.out
#SBATCH --error=job_outputs/site_figure.%j.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
activate_env

python -u "$ANALYSIS/site_description_figure.py" render \
    --sites "${LEFT_SITE:-carolina_beach}" "${RIGHT_SITE:-down_east}" \
    --layers_dir "$REPO_ROOT/data/site_figure/layers" \
    --grid_dir "$REPO_ROOT/data/grids" \
    --data_root "$REPO_ROOT/data" \
    --out_dir "$REPO_ROOT/data/site_figure" \
    "$@"
