#!/bin/bash
#SBATCH --job-name=my_jupyter_notebook
#SBATCH --account=st-jzhu71-1-gpu
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=gpu_mem_32
module load cuda/12.4.0 intel-oneapi-compilers/2023.1.0 python/3.11.6 gcc
export ENVDIR=/scratch/st-jzhu71-1/shenranw/envs/CoT     # change accordingly
source $ENVDIR/bin/activate

export HF_HOME="/scratch/st-jzhu71-1/shenranw/transformers_cache"
export INPUT="What is 103 times 202?"
export MODEL="hymba-1.5b-instruct"
export FILENAME="generation_cot_2.txt"

cd /scratch/st-jzhu71-1/shenranw/CoT
python ./generate.py