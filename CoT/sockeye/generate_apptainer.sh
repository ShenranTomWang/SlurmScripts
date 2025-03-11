#!/bin/bash
#SBATCH --job-name=my_jupyter_notebook
#SBATCH --account=st-jzhu71-1-gpu
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=gpu_mem_32
module load gcc apptainer
APPTAINER_DIR=/home/shenranw/envs/hymba.sif

apptainer exec --nv --fakeroot --home /scratch/st-jzhu71-1/shenranw/my_jupyter --env $APPTAINER_DIR ./apptainer_exec/generate.sh
