#!/bin/bash
 
#SBATCH --job-name=my_jupyter_notebook
#SBATCH --account=st-jzhu71-1-gpu
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=gpu_mem_32
module load gcc
module load apptainer

export MODEL="/home/shenranw/CoT/models/google/gemma-2-2b-it"
cd /scratch/st-jzhu71-1/shenranw/Hallucination
export START_IDX=50
export END_IDX=100
export DATASET="UMWPDataset"
echo $DATASET
export PROBE=1
apptainer run --nv --home /scratch/st-jzhu71-1/shenranw/Hallucination --env XDG_CACHE_HOME=$SLURM_SUBMIT_DIR /home/shenranw/jupyter/jupyter-datascience.sif python ./data_collection.py

export START_IDX=50
export END_IDX=100
export DATASET="UMWPDataset"
export PROBE=0
apptainer run --nv --home /scratch/st-jzhu71-1/shenranw/Hallucination --env XDG_CACHE_HOME=$SLURM_SUBMIT_DIR /home/shenranw/jupyter/jupyter-datascience.sif python ./data_collection.py

# export MODEL="/home/shenranw/CoT/models/google/gemma-2-2b-it"
# cd /scratch/st-jzhu71-1/shenranw/Hallucination
# export START_IDX=0
# export END_IDX=-1
# export DATASET="TruthfulQA"
# export PROBE=1
# apptainer run --nv --home /scratch/st-jzhu71-1/shenranw/Hallucination --env XDG_CACHE_HOME=$SLURM_SUBMIT_DIR /home/shenranw/jupyter/jupyter-datascience.sif python ./data_collection.py

# export START_IDX=0
# export END_IDX=-1
# export DATASET="TruthfulQA"
# export PROBE=0
# apptainer run --nv --home /scratch/st-jzhu71-1/shenranw/Hallucination --env XDG_CACHE_HOME=$SLURM_SUBMIT_DIR /home/shenranw/jupyter/jupyter-datascience.sif python ./data_collection.py
