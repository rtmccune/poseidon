#!/bin/bash
# Shared setup sourced by every POSEIDON submission script.
#
# Works under both schedulers while the HPC moves from LSF to Slurm.
# Submit scripts from this folder: `sbatch 02_segment_lists.sh` for
# scripts with #SBATCH lines (the GPU stages), `bsub < 05_...sh` for
# scripts with #BSUB lines.
# Sourcing this file:
#   * loads ~/.bashrc (for conda) and ../hpc_paths.env
#   * defines SUBMIT_DIR, REPO_ROOT, NAIADS, ANALYSIS, BIN_DIR
#   * defines TASK_ID (this array task's 1-based index, empty outside
#     an array) and NPROC (CPUs given to the job) for either scheduler
#   * defines SITE_DATA=$REPO_ROOT/data/$SITE and EVENT_DIR when SITE is
#     set in hpc_paths.env
#   * creates job_outputs/ for array-job logs
# and provides the helper functions below.

source ~/.bashrc
set -e  # stop the job at the first failing command

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${LS_SUBCWD:-$PWD}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-$LSB_JOBINDEX}"
NPROC="${SLURM_CPUS_PER_TASK:-${SLURM_NTASKS:-${LSB_DJOB_NUMPROC:-1}}}"
REPO_ROOT="$(cd "$SUBMIT_DIR/../.." && pwd)"
NAIADS="$REPO_ROOT/poseidon_deploy/naiads"
ANALYSIS="$REPO_ROOT/poseidon_deploy/analysis"
BIN_DIR="$REPO_ROOT/poseidon_utils/bin"
mkdir -p "$SUBMIT_DIR/job_outputs"

ENV_FILE="$SUBMIT_DIR/../hpc_paths.env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo "Loaded $ENV_FILE"
else
    echo "Warning: $ENV_FILE not found (copy hpc_paths.template.env)."
fi

if [ -n "$SITE" ]; then
    SITE_DATA="$REPO_ROOT/data/$SITE"
    EVENT_DIR="$SITE_DATA/flood_events"
fi

# Print a message to stderr and stop the job.
die() {
    echo "ERROR: $*" >&2
    exit 1
}

# Activate the conda environment named by POSEIDON_ENV.
activate_env() {
    [ -n "$POSEIDON_ENV" ] || die "POSEIDON_ENV is not set in hpc_paths.env"
    conda activate "$POSEIDON_ENV"
    echo "Activated conda environment $POSEIDON_ENV"
}

# Stop unless SITE names a config in poseidon_deploy/configs/sites.
require_site() {
    [ -n "$SITE" ] || die "SITE is not set in hpc_paths.env"
    local config="$REPO_ROOT/poseidon_deploy/configs/sites/$SITE.yaml"
    [ -f "$config" ] || die "No site config for '$SITE': $config"
    echo "Site: $SITE"
}

# Echo the event folder for this array task.
# Uses $TARGET_EVENT (a single event folder name) when set, otherwise
# TASK_ID-th (1-based) sorted subfolder of $EVENT_DIR.
event_for_task() {
    local event
    if [ -n "$TARGET_EVENT" ]; then
        event="$EVENT_DIR/$TARGET_EVENT"
    else
        [ -n "$TASK_ID" ] || die "Not an array job; set TARGET_EVENT"
        local events=($(find "$EVENT_DIR" -mindepth 1 -maxdepth 1 -type d \
            | sort))
        event="${events[$((TASK_ID - 1))]}"
    fi
    [ -d "$event" ] || die "Event folder not found: $event"
    echo "$event"
}

# Echo this array task's file list inside folder $1.
file_list_for_task() {
    local list="$1/file_list_${TASK_ID}.txt"
    [ -f "$list" ] || die "File list not found: $list"
    echo "$list"
}
