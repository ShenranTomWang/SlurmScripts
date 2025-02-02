#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --account=def-jjnunez

module load intel-oneapi-compilers/2023.1.0 python/3.11.6 gcc
export ENVDIR=/scratch/st-jjnunez-1/shenranw/envs/Interpretation     # change accordingly
python -m venv $ENVDIR
source $ENVDIR/bin/activate     # activate the virtual environment
export MPLCONFIGDIR=/scratch/st-jjnunez-1/shenranw/
export TRITON_CACHE_DIR="/scratch/st-jzhu71-1/shenranw/triton_cache"
export HF_HOME="/scratch/st-jzhu71-1/shenranw/transformers_cache"

cd /scratch/st-jzhu71-1/shenranw/Interpretation
python ./bbc_classification.py