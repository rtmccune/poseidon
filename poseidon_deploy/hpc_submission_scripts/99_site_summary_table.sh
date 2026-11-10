#!/bin/bash
# Analysis: paper table of flooding at each site (in-situ and images).
#
# Slurm (CPU partner nodes): submit with `sbatch 99_site_summary_table.sh`
# after stage 10 has run for both sites. Ignores SITE in hpc_paths.env;
# set LEFT_SITE / RIGHT_SITE to change the columns. Options after the
# script name go to site_summary_table.py, e.g.
#   sbatch 99_site_summary_table.sh --max_gap_minutes 15
#
# Input:  data/<site>/flood_events.csv and data/<site>/flood_events/
#         (<roi>_accessibility_time_series.csv from stage 10)
# Output: data/publication_combined/site_summary_table.{png,pdf,svg,csv,
#         tex,md} from every image, including events without sensor
#         data, and site_summary_table_good_only.* from well-segmented
#         images only, for comparison; the .md defines every row and how
#         they differ from stage 11's summary
#
#SBATCH --job-name=site_table
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=kanarde_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --output=job_outputs/site_table.%j.out
#SBATCH --error=job_outputs/site_table.%j.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
activate_env

python -u "$ANALYSIS/site_summary_table.py" \
    --sites "${LEFT_SITE:-carolina_beach}" "${RIGHT_SITE:-down_east}" \
    --data_root "$REPO_ROOT/data" \
    --out_dir "$REPO_ROOT/data/publication_combined" \
    "$@"
