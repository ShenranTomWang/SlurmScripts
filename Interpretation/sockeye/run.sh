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

export CONFIG="configs/bbc_config.json"
export MODEL="/scratch/st-jjnunez-1/shenranw/models/Umesh/distilbert-bbc-news-classification"
python ./run.py --config $CONFIG --model $MODEL --multiclass huggingface --model_class DistilBertBBCNewsClassifier

export CONFIG="configs/topics_classification-distilroberta.json"
export MODEL="/scratch/st-jjnunez-1/shenranw/models/valurank/distilroberta-topic-classification"
python ./run.py --config $CONFIG --model $MODEL --multiclass huggingface --model_class DistilRobertaTopicsClassifier

export CONFIG="configs/topics_classification-gpt2.json"
export MODEL="/scratch/st-jjnunez-1/shenranw/models/openai-community/gpt2"
python ./run.py --config $CONFIG --model $MODEL --multiclass llm_classifier_model --llm_classifier_config_dir "out/gpt2/config.json" --llm_classifier_dir "out/gpt2/classifier.pth"

export CONFIG="configs/topics_classification-Llama-3.2-1B.json"
export MODEL="/scratch/st-jjnunez-1/shenranw/models/meta-llama/Llama-3.2-1B"
python ./run.py --config $CONFIG --model $MODEL --multiclass llm_classifier_model --llm_classifier_config_dir "out/Llama-3.2-1B/config.json" --llm_classifier_dir "out/Llama-3.2-1B/classifier.pth"