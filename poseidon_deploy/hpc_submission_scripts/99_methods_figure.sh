#!/bin/bash
# Analysis: example images and process diagram for the methods figure.
#
# Slurm (CPU partner nodes): submit with `sbatch 99_methods_figure.sh`
# after stages 08-10 have run for the site (SITE in hpc_paths.env).
# The example image is chosen automatically (well segmented, roadway
# coverage near 60%); pick another with options after the script name,
# which go to methods_figure_assets.py, e.g.
#   sbatch 99_methods_figure.sh --target_coverage 40
#   sbatch 99_methods_figure.sh --event <event folder> --timestamp <14 digits>
#
# Output: $SITE_DATA/methods_figure/assets/ (panels and manifest.json)
#         $SITE_DATA/methods_figure/process_diagram.svg (links the
#         panels; open in Inkscape from that folder)
#
#SBATCH --job-name=methods_fig
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --account=kanarde_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --output=job_outputs/methods_fig.%j.out
#SBATCH --error=job_outputs/methods_fig.%j.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
activate_env

OUT="$SITE_DATA/methods_figure"
python -u "$ANALYSIS/methods_figure_assets.py" \
    --site "$SITE" \
    --events_dir "$EVENT_DIR" \
    --images_dir "$SITE_DATA/images" \
    --labels "$SITE_DATA/segmentation_labels.csv" \
    --sensor "$SITE_DATA/sensor_data_comparison.csv" \
    --grid "$REPO_ROOT/data/grids/shared_grid_$SITE.npz" \
    --out_dir "$OUT/assets" \
    "$@"

python -u "$ANALYSIS/methods_process_diagram.py" \
    --site "$SITE" \
    --assets "$OUT/assets" \
    --out "$OUT/process_diagram.svg"
