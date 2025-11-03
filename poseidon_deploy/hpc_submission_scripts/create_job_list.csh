#! /bin/bash
#BSUB -J list_steves
#BSUB -o job_list.%J.out
#BSUB -e job_list.%J.err
#BSUB -W 60
#BSUB -n 20
#BSUB -q ccee

source ~/.bashrc

module load PrgEnv-intel
conda activate /rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/conda/image_processing
python prepare_job_lists.py /share/jcdietri/rmccune/depth_map/data/down_east/predsegs 37

conda deactivate
