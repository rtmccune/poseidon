#! /bin/bash
#BSUB -J seg_model
#BSUB -o preds_out.%J
#BSUB -e preds_err.%J
#BSUB -W 360
#BSUB -n 1
#BSUB -R span[hosts=1]
#BSUB -R rusage[mem=16G]
#BSUB -q gpu
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[a100 || l40 || l40s || h100]"

source ~/.bashrc

export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_CACHE="/share/jcdietri/rmccune/segmentation/segmentation_gym/hf_cache_portable"

module load cuda/12.1
module load apptainer

# Define the project base directory to make binding easier
PROJECT_DIR="/share/jcdietri/rmccune/segmentation"

# Define the path to your Apptainer image
IMAGE_PATH="${PROJECT_DIR}/training/container/seg_gym_tf.sif"

# Define the name of the specific data directory you wish to use
DATA_DIR_NAME="all_sites" ####---EDIT THIS LINE---####
IMAGES_DIR_NAME="/share/jcdietri/rmccune/poseidon/data/down_east/images/daylight_all_events" ####---EDIT THIS LINE---####

# Define weights file directory
WEIGHTS_DIR="${PROJECT_DIR}/data/${DATA_DIR_NAME}/weights"

# Execute the container with the correct syntax
apptainer exec --nv \
    --bind /share/jcdietri/rmccune:/share/jcdietri/rmccune \
    ${IMAGE_PATH} \
    python ${PROJECT_DIR}/segmentation_gym/seg_images_in_folder_no_tkinter.py \
    --images_dir ${IMAGES_DIR_NAME} \
    --weights ${WEIGHTS_DIR}/all_sites_5_class_v3_segformer_fullmodel.h5 ####---EDIT THIS LINE---####
