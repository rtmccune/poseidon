#! /bin/bash
#BSUB -J seg_batch[1-32] ####--- EDIT THIS to match your number of file lists ---####
#BSUB -W 45
#BSUB -n 1
#BSUB -R "rusage[mem=4G]"
#BSUB -gpu "num=1:mode=shared"
#BSUB -q gpu
#BSUB -o job_outputs/seg_batch.%J.%I.out
#BSUB -e job_outputs/seg_batch.%J.%I.err

source ~/.bashrc

module load cuda/12.1
module load apptainer

# Resolve the directory where the job was submitted from (LSF variable or fallback)
SUBMIT_DIR="${LS_SUBCWD:-$PWD}"

# Define the project base directory to make binding easier
PROJECT_DIR="$SUBMIT_DIR/../.."

# Create output directory for logs if it doesn't exist
mkdir -p $SUBMIT_DIR/job_outputs

export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_CACHE="$PROJECT_DIR/poseidon_deploy/segmentation/segmentation_gym/hf_cache_portable"

# --- CONFIGURATION ---
CONTAINER_PATH="${PROJECT_DIR}/poseidon_deploy/segmentation/container/seg_gym.sif"

WEIGHTS_FILE="${PROJECT_DIR}/data/segmentation/all_sites/weights/all_sites_5_class_v3_segformer_fullmodel.h5"

# Directory containing your images (still needed by the script for output structure)
IMAGES_DIR_NAME="${PROJECT_DIR}/data/down_east/images/daylight_all_events"

# Directory where you saved your text file lists from Step 2
LISTS_DIR="${IMAGES_DIR_NAME}/job_file_lists"

# Determine which file list this specific task should process.
# LSF uses %I for index in filenames, but $LSB_JOBINDEX in the script.
# If you named them file_list_00.txt, file_list_01.txt, use printf to pad zeros if needed.
# For simple 1, 2, 3 indexing:
FILE_LIST="${LISTS_DIR}/file_list_${LSB_JOBINDEX}.txt"

# (Alternative: if you used split -d and have 00, 01, 02... you might need formatting):
# INDEX=$(printf "%02d" $((LSB_JOBINDEX - 1)))
# FILE_LIST="${LISTS_DIR}/file_list_${INDEX}.txt"

echo "=================================================="
echo "Job Index: ${LSB_JOBINDEX}"
echo "Processing file list: ${FILE_LIST}"
echo "=================================================="

if [ ! -f "${FILE_LIST}" ]; then
    echo "ERROR: File list not found: ${FILE_LIST}"
    exit 1
fi

# Execute the container
apptainer exec --nv \
    --bind /share/jcdietri/rmccune:/share/jcdietri/rmccune \
    ${CONTAINER_PATH} \
    python ${PROJECT_DIR}/poseidon_deploy/segmentation/segmentation_gym/seg_images_in_folder_no_tkinter.py \
    --images_dir ${IMAGES_DIR_NAME} \
    --weights ${WEIGHTS_FILE} \
    --file_list ${FILE_LIST}
