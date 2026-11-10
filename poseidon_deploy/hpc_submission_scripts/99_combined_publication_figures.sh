#!/bin/bash
# Analysis: two-site publication figures, one site per side.
#
# Slurm (CPU partner nodes): submit with
#   sbatch 99_combined_publication_figures.sh
# after stages 09 and 10 have run for both sites. Ignores SITE in
# hpc_paths.env; set LEFT_SITE / RIGHT_SITE to change the sites, e.g.
#   LEFT_SITE=down_east RIGHT_SITE=carolina_beach \
#       sbatch 99_combined_publication_figures.sh
#
# Input:  data/<site>/flood_events, sensor_data_comparison.csv and
#         segmentation_labels.csv for each site; ROI: each site's
#         camera_roi_json, else roi_json
# Output: data/publication_combined/combined_fig1_aligned_timing and
#         combined_fig2_depth_by_coverage (.png, .pdf), plus each site's
#         peak-aligned panel (a) means and 1c fit summary CSVs. See
#         analysis/combined_publication_figures.py.
#
#SBATCH --job-name=pub_combined
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --account=kanarde_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --output=job_outputs/pub_combined.%j.out
#SBATCH --error=job_outputs/pub_combined.%j.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
activate_env

python -u "$ANALYSIS/combined_publication_figures.py" \
    --sites "${LEFT_SITE:-carolina_beach}" "${RIGHT_SITE:-down_east}" \
    --data_root "$REPO_ROOT/data" \
    --out_dir "$REPO_ROOT/data/publication_combined"
