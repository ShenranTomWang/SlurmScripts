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

cd /scratch/st-jjnunez-1/shenranw/Interpretation

export MODEL="/scratch/st-jjnunez-1/shenranw/models/Umesh/distilbert-bbc-news-classification"
python ./test.py --model $MODEL --data_path "data/bbc_news/distilbert-bbc-news-classification/test.tsv" --dataset "bbc_news"

export MODEL="/scratch/st-jjnunez-1/shenranw/models/openai-community/gpt2"
python ./test.py --model $MODEL --data_path "data/topics_classification/gpt2-topics-classification/test.tsv" --dataset "topics_classification"

export MODEL="/scratch/st-jjnunez-1/shenranw/models/meta-llama/Llama-3.2-1B"
python ./test.py --model $MODEL --data_path "data/topics_classification/Llama-3.2-1B-topics-classification/test.tsv" --dataset "topics_classification"

export MODEL="/scratch/st-jjnunez-1/shenranw/models/valurank/distilroberta-topic-classification"
python ./test.py --model $MODEL --data_path "data/topics_classification/distilroberta-topics-classification/test.tsv" --dataset "topics_classification"