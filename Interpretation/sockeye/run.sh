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
export TRITON_CACHE_DIR="/scratch/st-jzhu71-1/shenranw/triton_cache"
export HF_HOME="/scratch/st-jzhu71-1/shenranw/transformers_cache"

cd /scratch/st-jzhu71-1/shenranw/Interpretation

export CONFIG="configs/bbc_config.json"
export MODEL="/scratch/st-jzhu71-1/shenranw/models/Umesh/distilbert-bbc-news-classification"
python ./run.py --config $CONFIG --model $MODEL --multiclass huggingface

export CONFIG="configs/topics_classification.json"
export MODEL="/scratch/st-jzhu71-1/shenranw/models/openai-community/gpt2"
python ./run.py --config $CONFIG --model $MODEL --multiclass llm_classifier_model --llm_classifier_config_dir "out/gpt2/config.json" --llm_classifier_dir "out/gpt2/classifier.pth"