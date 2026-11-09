#!/bin/bash
# Analysis: compare the in-situ sensor with camera-derived depths (virtual
# sensors and the roadway ROI) across all flood events.
#
# Slurm (CPU partner nodes): submit with `sbatch 99_sensor_camera_comparison.sh`
# after stages 09 (virtual sensors) and 10 (ROI statistics).
#
# Input:  $EVENT_DIR/<event>/zarr/virtual_sensor_data.zip and
#         <roi>_accessibility_time_series.csv (ROI: the site's
#         camera_roi_json, else roi_json)
#         $SITE_DATA/sensor_data_comparison.csv (in-situ readings)
#         $SITE_DATA/segmentation_labels.csv (adds the good_only set)
# Output: $SITE_DATA/plots/<roi>_sensor_camera_<set>_{depths,timing,peaks}.jpg,
#         CSVs, and <roi>_sensor_camera_<set>_summary.md, where <set> is
#         all_images and (with labels) good_only
#         (see analysis/sensor_camera_comparison.py for every definition)
#
#SBATCH --job-name=sensor_camera
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=kanarde_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --output=job_outputs/sensor_camera.%j.out
#SBATCH --error=job_outputs/sensor_camera.%j.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
activate_env

python -u "$ANALYSIS/sensor_camera_comparison.py" \
    --site "$SITE" \
    --events_dir "$EVENT_DIR" \
    --sensor "$SITE_DATA/sensor_data_comparison.csv" \
    --labels "$SITE_DATA/segmentation_labels.csv" \
    --out_dir "$SITE_DATA/plots"
