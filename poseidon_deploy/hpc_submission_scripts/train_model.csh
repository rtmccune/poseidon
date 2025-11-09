#! /bin/bash

#BSUB -J train_model
#BSUB -W 420
#BSUB -n 12
#BSUB -R rusage[mem=20G]
#BSUB -R "select[a100 || l40 || l40s || h100]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpu
#BSUB -o train_model.%J.out
#BSUB -e train_model.%J.err

source ~/.bashrc

module load cuda/12.1
module load apptainer

# Resolve the directory where the job was submitted from (LSF variable or fallback)
SUBMIT_DIR="${LS_SUBCWD:-$PWD}"

# Define the project base directory to make binding easier
PROJECT_DIR="$SUBMIT_DIR/../.."

export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_CACHE="$PROJECT_DIR/poseidon_deploy/segmentation/segmentation_gym/hf_cache_portable"

# Define the path to your Apptainer image
CONTAINER_PATH="${PROJECT_DIR}/poseidon_deploy/segmentation/container/seg_gym.sif"

# Define the name of the specific data directory you wish to use
DATA_DIR_NAME="$PROJECT_DIR/data/all_sites" ####---EDIT THIS LINE---####

# Define NPZ and configuration file directories
NPZ_DIR="${DATA_DIR_NAME}/fromDoodler/npz4gym"
CONFIG_DIR="${DATA_DIR_NAME}/config"

# Execute the container with the correct syntax
apptainer exec --nv \
    --bind ${PROJECT_DIR} \
    ${CONTAINER_PATH} \
    python ${PROJECT_DIR}/poseidon_deploy/segmentation/segmentation_gym/train_model_script_no_tkinter.py \
    --train_data_dir ${NPZ_DIR}/train_data/train_npzs \
    --val_data_dir ${NPZ_DIR}/val_data/val_npzs \
    --config_file ${CONFIG_DIR}/all_sites_5_class_v4_segformer.json ####---EDIT THIS LINE---####
