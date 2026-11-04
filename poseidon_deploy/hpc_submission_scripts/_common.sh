#!/bin/bash
# Shared setup sourced by every POSEIDON LSF submission script.
#
# Submit scripts from this folder (e.g. `bsub < 07_make_rectify.sh`).
# Sourcing this file:
#   * loads ~/.bashrc (for conda) and ../hpc_paths.env
#   * defines SUBMIT_DIR, REPO_ROOT, NAIADS, ANALYSIS, BIN_DIR
#   * defines SITE_DATA=$REPO_ROOT/data/$SITE and EVENT_DIR when SITE is
#     set in hpc_paths.env
#   * creates job_outputs/ for array-job logs
# and provides the helper functions below.

source ~/.bashrc
set -e  # stop the job at the first failing command

SUBMIT_DIR="${LS_SUBCWD:-$PWD}"
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
# the LSB_JOBINDEX-th (1-based) sorted subfolder of $EVENT_DIR.
event_for_task() {
    local event
    if [ -n "$TARGET_EVENT" ]; then
        event="$EVENT_DIR/$TARGET_EVENT"
    else
        [ -n "$LSB_JOBINDEX" ] || die "Not an array job; set TARGET_EVENT"
        local events=($(find "$EVENT_DIR" -mindepth 1 -maxdepth 1 -type d \
            | sort))
        event="${events[$((LSB_JOBINDEX - 1))]}"
    fi
    [ -d "$event" ] || die "Event folder not found: $event"
    echo "$event"
}

# Echo this array task's file list inside folder $1.
file_list_for_task() {
    local list="$1/file_list_${LSB_JOBINDEX}.txt"
    [ -f "$list" ] || die "File list not found: $list"
    echo "$list"
}
