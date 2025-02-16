#!/bin/bash
#SBATCH --job-name=extract_activations
#SBATCH --account=st-jzhu71-1-gpu
#SBATCH --time=5-00:00:00
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

cd /scratch/st-jzhu71-1/shenranw/ICL

export MODEL="/scratch/st-jzhu71-1/shenranw/models/openai-community/gpt2"
export OUT_DIR="out/gpt2"
python extract_activations.py --model $MODEL --add_newlines --task class_to_class --k 4 --split demo --out_dir $OUT_DIR
python extract_activations.py --model $MODEL --add_newlines --task class_to_class_random --k 4 --split demo --out_dir $OUT_DIR

export MODEL="/scratch/st-jzhu71-1/shenranw/models/meta-llama/Llama-3.2-1B"
export OUT_DIR="out/Llama-3.2-1B"
python extract_activations.py --model $MODEL --add_newlines --task class_to_class --k 4 --split demo --out_dir $OUT_DIR
python extract_activations.py --model $MODEL --add_newlines --task class_to_class_random --k 4 --split demo --out_dir $OUT_DIR

export MODEL="/scratch/st-jzhu71-1/shenranw/models/Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5B"
python extract_activations.py --model $MODEL --add_newlines --task class_to_class --k 4 --split demo --out_dir $OUT_DIR
python extract_activations.py --model $MODEL --add_newlines --task class_to_class_random --k 4 --split demo --out_dir $OUT_DIR

export MODEL="/scratch/st-jzhu71-1/shenranw/models/nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python extract_activations.py --model $MODEL --add_newlines --task class_to_class --k 4 --split demo --out_dir $OUT_DIR
python extract_activations.py --model $MODEL --add_newlines --task class_to_class_random --k 4 --split demo --out_dir $OUT_DIR

export MODEL="/scratch/st-jzhu71-1/shenranw/models/RWKV/v6-Finch-1B6-HF"
export OUT_DIR="out/rwkv-v6-Finch-1B6-HF"
python extract_activations.py --model $MODEL --add_newlines --task class_to_class --k 4 --split demo --out_dir $OUT_DIR
python extract_activations.py --model $MODEL --add_newlines --task class_to_class_random --k 4 --split demo --out_dir $OUT_DIR