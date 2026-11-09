#!/bin/bash
# Stage 11: summarize roadway accessibility across all events.
#
# Slurm (CPU partner nodes): submit with `sbatch 11_summarize_road_stats.sh`.
# Output: summaries in the job log and
#         $SITE_DATA/road_access_summary.csv (roi_json)
# and, if the site config sets camera_roi_json:
#         $SITE_DATA/road_access_summary_camera.csv
#         $SITE_DATA/roi_comparison/ (plan-view vs camera-view ROI;
#         see analysis/compare_roadway_rois.py)
#
#SBATCH --job-name=stats
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=kanarde_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --output=summary_stats.%j.out
#SBATCH --error=summary_stats.%j.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
export TMPDIR=/tmp
activate_env

python -u "$NAIADS/run_road_access_stats.py" \
    --site "$SITE" \
    --event_dir "$EVENT_DIR" \
    --output_csv "$SITE_DATA/road_access_summary.csv"

if site_has camera_roi_json; then
    python -u "$NAIADS/run_road_access_stats.py" \
        --site "$SITE" \
        --roi_view camera \
        --event_dir "$EVENT_DIR" \
        --output_csv "$SITE_DATA/road_access_summary_camera.csv"

    python -u "$ANALYSIS/compare_roadway_rois.py" \
        --site "$SITE" \
        --events_dir "$EVENT_DIR" \
        --output_dir "$SITE_DATA/roi_comparison"
fi
