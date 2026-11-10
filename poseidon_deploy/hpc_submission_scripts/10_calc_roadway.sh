#!/bin/bash
# Stage 10: summarize roadway flooding per event (one task per event).
#
# Slurm (CPU partner nodes): submit with `sbatch 10_calc_roadway.sh`.
# Set --array to the number of event folders. To rerun one event:
#   TARGET_EVENT=<event folder name> sbatch --array=1 10_calc_roadway.sh
#
# Analyzes the site config's roi_json (drawn on a rectified image) and,
# if the config sets camera_roi_json, also that polygon drawn on a
# camera image, rectified with the site's camera model and grid. Both
# shapes must be labeled $ROI_LABEL.
# Output: $EVENT_DIR/<event>/<roi>_{accessibility,wse}_time_series.csv
#         and <roi>_roi_mask.png for each ROI
#
#SBATCH --job-name=roadway_batch
#SBATCH --array=1-82
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=kanarde_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --output=job_outputs/roadway.%A.%a.out
#SBATCH --error=job_outputs/roadway.%A.%a.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
require_site
export TMPDIR=/tmp
activate_env

ROI_LABEL="${ROI_LABEL:-roadway}"
EVENT=$(event_for_task)
echo "Task ${TASK_ID:-single}: $EVENT"

python -u "$NAIADS/run_roadway_analyzer.py" \
    --site "$SITE" \
    --target_event_dir "$EVENT" \
    --label "$ROI_LABEL" \
    --step_size 1.0 \
    --statistic 95_perc

if site_has camera_roi_json; then
    python -u "$NAIADS/run_roadway_analyzer.py" \
        --site "$SITE" \
        --roi_view camera \
        --grid_dir "$REPO_ROOT/data/grids" \
        --target_event_dir "$EVENT" \
        --label "$ROI_LABEL" \
        --statistic 95_perc
fi
