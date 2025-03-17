#!/bin/bash
#SBATCH --job-name=bbc_classification
#SBATCH --account=st-jjnunez-1-gpu
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1

module load intel-oneapi-compilers/2023.1.0 python/3.11.6 gcc
export ENVDIR=/scratch/st-jjnunez-1/shenranw/envs/Interpretation     # change accordingly
source $ENVDIR/bin/activate     # activate the virtual environment
export MPLCONFIGDIR="/scratch/st-jjnunez-1/shenranw/"
export TRITON_CACHE_DIR="/scratch/st-jjnunez-1/shenranw/triton_cache"
export HF_HOME="/scratch/st-jjnunez-1/shenranw/transformers_cache"

cd /scratch/st-jjnunez-1/shenranw/Interpretation/models/llm

export CONFIG_DIR="/scratch/st-jjnunez-1/shenranw/Interpretation/data/topics_classification/Llama-3.2-1B-topics-classification"
export OUT_DIR="/scratch/st-jjnunez-1/shenranw/Intepretation/out/Llama-3.2-1B"
export HIDDEN_DIMS=2048
python ./train_classifier.py --config_dir $CONFIG_DIR --out_dir $OUT_DIR --hidden_dims $HIDDEN_DIMS

export CONFIG_DIR="/scratch/st-jjnunez-1/shenranw/Interpretation/data/topics_classification/gpt2-topics-classification"
export OUT_DIR="/scratch/st-jjnunez-1/shenranw/Intepretation/out/gpt2"
export HIDDEN_DIMS=768
python ./train_classifier.py --config_dir $CONFIG_DIR --out_dir $OUT_DIR --hidden_dims $HIDDEN_DIMS