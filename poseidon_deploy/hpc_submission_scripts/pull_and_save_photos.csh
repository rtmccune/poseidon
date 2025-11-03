#! /bin/bash

#BSUB -J photo_pull
#BSUB -o photo_pull_out.%J
#BSUB -e photo_pull_err.%J
#BSUB -W 120
#BSUB -n 1
#BSUB -q ccee

echo "Activating conda environment..."
source ~/.bashrc
conda activate /rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/conda/image_processing

echo "Starting photo pull Python script..."
python save_photos.py

echo "Deactivating conda environment..."
conda deactivate

echo "Job finished."