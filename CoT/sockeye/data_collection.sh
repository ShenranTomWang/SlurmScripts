#!/bin/bash
#SBATCH --job-name=my_jupyter_notebook
#SBATCH --account=st-jzhu71-1-gpu
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=gpu_mem_32
module load intel-oneapi-compilers/2023.1.0 python/3.11.6 gcc
export ENVDIR=/scratch/st-jzhu71-1/shenranw/envs/CoT     # change accordingly
source $ENVDIR/bin/activate
cd /scratch/st-jzhu71-1/shenranw/CoT

export TRITON_CACHE_DIR="/scratch/st-jzhu71-1/shenranw/triton_cache"
export HF_HOME="/scratch/st-jzhu71-1/shenranw/transformers_cache"

export DATASET="GSM8K"
export MAX_IDX=200
export MODEL="gemma-2-2b-it"

python ./data_collection.py