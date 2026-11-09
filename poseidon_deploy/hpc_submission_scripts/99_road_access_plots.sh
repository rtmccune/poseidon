#!/bin/bash
# Analysis: plot roadway accessibility statistics across all events.
#
# Slurm (CPU partner nodes): submit with `sbatch 99_road_access_plots.sh`
# after stage 10. Plots every ROI found in the event folders (e.g. both
# roi_json and camera_roi_json).
#
# Input:  $EVENT_DIR/<event>/<roi>_*_time_series.csv
#         $SITE_DATA/segmentation_labels.csv (Filename, Segmentation_Score)
#             marks well-segmented images (blue); without it, the
#             good-segmentation and sensor plots are skipped
#         $SITE_DATA/sensor_data_comparison.csv (optional) in-situ
#             readings, e.g. from naiads/run_pull_event_series.py with
#             --buffer_hours 3 --output $SITE_DATA/sensor_data_comparison.csv
# Output: $SITE_DATA/plots/, per ROI and depth metric:
#         *_1c_binned_depth_good_only.jpg and *_1c_depth_by_coverage.csv
#             depth by coverage bin, with a separate full-coverage bin;
#             the CSV is the depth lookup table (1b: all images)
#         *_1c_reversed_fit_good_only.jpg with *_fit_summary.csv and
#             *_band_calibration.csv (logistic summary and its checks)
#         event-peak plots 2-5 and sensor comparisons 6-7
#         See analysis/access_plotter.py for every definition.
#
#SBATCH --job-name=plot_access
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=kanarde_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --output=job_outputs/poseidon_plots.%j.out
#SBATCH --error=job_outputs/poseidon_plots.%j.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
activate_env

python -u "$ANALYSIS/access_plotter.py" \
    --site "$SITE" \
    --events_dir "$EVENT_DIR" \
    --out_dir "$SITE_DATA/plots" \
    --labels "$SITE_DATA/segmentation_labels.csv" \
    --sensor "$SITE_DATA/sensor_data_comparison.csv"
