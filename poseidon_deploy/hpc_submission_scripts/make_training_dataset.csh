#! /bin/bash

#BSUB -J make_data
#BSUB -W 60
#BSUB -n 12
#BSUB -R rusage[mem=12G]
#BSUB -gpu "num=1"
#BSUB -q gpu
#BSUB -o make_data.%J.out
#BSUB -e make_data.%J.err

source ~/.bashrc

module load cuda/12.1
module load apptainer

# Resolve the directory where the job was submitted from (LSF variable or fallback)
SUBMIT_DIR="${LS_SUBCWD:-$PWD}"

# Define the project base directory to make binding easier
PROJECT_DIR="$SUBMIT_DIR/../.."

# Define the path to your Apptainer image
CONTAINER_PATH="${PROJECT_DIR}/poseidon_deploy/segmentation/container/seg_gym.sif"

# Define the name of the specific data directory you wish to use
DATA_DIR_NAME="$PROJECT_DIR/data/all_sites" ####---EDIT THIS LINE---####

CONFIG_DIR="${PROJECT_DIR}/data/${DATA_DIR_NAME}/config"

# Execute the container with the correct syntax
apptainer exec --nv \
    --bind ${PROJECT_DIR} \
    ${CONTAINER_PATH} \
    python ${PROJECT_DIR}/poseidon_deploy/segmentation/segmentation_gym/make_dataset_no_tkinter_updated.py \
    --output ${DATA_DIR_NAME}/fromDoodler/npz4gym \
    --label_dir ${DATA_DIR_NAME}/fromDoodler/labels \
    --image_dirs ${PROJECT_DIR}/data/${DATA_DIR_NAME}/fromDoodler/images \
    --config ${DATA_DIR_NAME}/config/all_sites_5_class_v4_segformer.json ###---EDIT THIS LINE---###
