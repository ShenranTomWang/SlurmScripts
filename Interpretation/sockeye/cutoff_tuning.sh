#!/bin/bash
#SBATCH --job-name=cutoff_tuning
#SBATCH --account=st-jjnunez-1-gpu
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1

module load intel-oneapi-compilers/2023.1.0 python/3.11.6 gcc
export ENVDIR=/scratch/st-jjnunez-1/shenranw/envs/Interpretation     # change accordingly
source $ENVDIR/bin/activate     # activate the virtual environment
export MPLCONFIGDIR=/scratch/st-jjnunez-1/shenranw/
export TRITON_CACHE_DIR="/scratch/st-jjnunez-1/shenranw/triton_cache"
export HF_HOME="/scratch/st-jjnunez-1/shenranw/transformers_cache"

cd /scratch/st-jjnunez-1/shenranw/Interpretation
python ./cutoff_tuning.py --config configs/bbc_config.json --out_dir results/bbc_news/distilbert-bbc-news-classification --evaluation cross_entropy