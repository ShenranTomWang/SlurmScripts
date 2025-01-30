#!/bin/bash
#SBATCH --job-name=ICL-gpt2
#SBATCH --account=st-jzhu71-1-gpu
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=gpu_mem_32

module load intel-oneapi-compilers/2023.1.0 python/3.11.6 gcc
module load cuda/12.4.0
export ENVDIR=/scratch/st-jzhu71-1/shenranw/envs/CoT     # change accordingly
source $ENVDIR/bin/activate

export TRITON_CACHE_DIR="/scratch/st-jzhu71-1/shenranw/triton_cache"
export HF_HOME="/scratch/st-jzhu71-1/shenranw/transformers_cache"

export MODEL="/scratch/st-jzhu71-1/shenranw/models/openai-community/gpt2"
export OUT_DIR="out/gpt2"

cd /scratch/st-jzhu71-1/shenranw/ICL
python test_custom.py --model $MODEL --add_newlines --dataset sms_spam_random --out_dir $OUT_DIR/sms_spam_random --k 4 --n_skips -1
python test_custom.py --model $MODEL --add_newlines --dataset sms_spam --out_dir $OUT_DIR/sms_spam --k 4 --n_skips -1