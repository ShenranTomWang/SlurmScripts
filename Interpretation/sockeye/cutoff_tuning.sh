#!/bin/bash
#SBATCH --job-name=ICL-gpt2
#SBATCH --account=st-jjnunez-1-gpu
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1

module load intel-oneapi-compilers/2023.1.0 python/3.11.6 gcc
export ENVDIR=/scratch/st-jjnunez-1/shenranw/envs/Interpretation     # change accordingly
python -m venv $ENVDIR
source $ENVDIR/bin/activate     # activate the virtual environment
export MPLCONFIGDIR=/scratch/st-jjnunez-1/shenranw/
export TRITON_CACHE_DIR="/scratch/st-jzhu71-1/shenranw/triton_cache"
export HF_HOME="/scratch/st-jzhu71-1/shenranw/transformers_cache"

cd /scratch/st-jzhu71-1/shenranw/Interpretation
python ./cutoff_tuning.py --config configs/bbc_config.json --out_dir results/bbc_news/distilbert-bbc-news-classification