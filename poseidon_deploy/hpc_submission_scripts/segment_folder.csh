#! /bin/bash
#BSUB -J seg_folder
#BSUB -W 360
#BSUB -n 1
#BSUB -R span[hosts=1]
#BSUB -R rusage[mem=4G]
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[a100 || l40 || l40s || h100]"
#BSUB -q gpu
#BSUB -o seg_folder.%J.out
#BSUB -e seg_folder.%J.err

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

IMAGES_DIR_NAME="${PROJECT_DIR}/data/down_east/images/daylight_all_events" ####---EDIT THIS LINE---####

WEIGHTS_FILE="${PROJECT_DIR}/data/segmentation/all_sites/weights/all_sites_5_class_v3_segformer_fullmodel.h5" ####---EDIT THIS LINE---####

# Execute the container with the correct syntax
apptainer exec --nv \
    --bind ${PROJECT_DIR} \
    ${CONTAINER_PATH} \
    python ${PROJECT_DIR}/poseidon_deploy/segmentation/segmentation_gym/seg_images_in_folder_no_tkinter.py \
    --images_dir ${IMAGES_DIR_NAME} \
    --weights ${WEIGHTS_FILE} 
