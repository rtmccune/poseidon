#!/bin/bash
# Analysis: example flood event figure per site (three panes above the
# water depth time series, at the rising leg, peak and falling leg).
#
# Slurm (CPU partner nodes): submit with `sbatch 99_example_event_figure.sh`
# after stages 08-10 have run for both sites. Ignores SITE in
# hpc_paths.env; set SITES to change them (default
# "carolina_beach down_east"). The event and moments are chosen
# automatically; options after the script name go to
# example_event_figure.py, e.g.
#   sbatch 99_example_event_figure.sh --pane image
#   sbatch 99_example_event_figure.sh --events <CB event> auto
#   sbatch 99_example_event_figure.sh --events <CB event> auto \
#       --times carolina_beach <rising> <peak> <falling>
# (timestamps are 14-digit UTC, as in the image filenames; the CSV of a
# first run lists the ones it chose).
#
# Output: data/publication_combined/example_event_<site>.{png,pdf,svg,csv}
#
#SBATCH --job-name=example_event
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=kanarde_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --output=job_outputs/example_event.%j.out
#SBATCH --error=job_outputs/example_event.%j.err

source "${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}/_common.sh"
activate_env

python -u "$ANALYSIS/example_event_figure.py" \
    --sites ${SITES:-carolina_beach down_east} \
    --data_root "$REPO_ROOT/data" \
    --out_dir "$REPO_ROOT/data/publication_combined" \
    "$@"
