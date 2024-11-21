#!/bin/bash
 
#SBATCH --job-name=my_jupyter_notebook
#SBATCH --account=st-jzhu71-1-gpu
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=gpu_mem_32
module load gcc
module load apptainer
apptainer run --nv --home /scratch/st-jzhu71-1/shenranw/my_jupyter --env XDG_CACHE_HOME=$SLURM_SUBMIT_DIR /home/shenranw/jupyter/jupyter-datascience.sif ./apptainer_run/data_collection.sh
