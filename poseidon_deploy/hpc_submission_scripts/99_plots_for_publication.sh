#!/bin/bash
# Analysis: make the publication figures (well-segmented images only).
#
# Slurm (CPU partner nodes): submit with `sbatch 99_plots_for_publication.sh`
# after stages 09 (virtual sensors) and 10 (ROI statistics).
#
# Input:  $EVENT_DIR, $SITE_DATA/sensor_data_comparison.csv,
#         $SITE_DATA/segmentation_labels.csv; ROI: the site's
#         camera_roi_json, else roi_json
# Output: $SITE_DATA/publication/<roi>_pub_fig1_aligned_timing.{png,pdf}
#         (event-aligned mean depths, VS1 vs in-situ sensor, onset lead
#         and end lag) and <roi>_pub_fig2_binned_depth.{png,pdf} (image
#         mean and maximum depth and sensor depth by coverage), with
#         CSVs of every number shown. See
#         analysis/publication_figures.py for every definition.
#
#SBATCH --job-name=pub_figures
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=kanarde_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --output=job_outputs/pub_figures.%j.out
#SBATCH --error=job_outputs/pub_figures.%j.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
activate_env

python -u "$ANALYSIS/publication_figures.py" \
    --site "$SITE" \
    --events_dir "$EVENT_DIR" \
    --sensor "$SITE_DATA/sensor_data_comparison.csv" \
    --labels "$SITE_DATA/segmentation_labels.csv" \
    --out_dir "$SITE_DATA/publication"
