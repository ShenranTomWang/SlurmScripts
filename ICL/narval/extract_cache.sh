#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --account=def-lingjzhu

export ENVDIR=/scratch/shenranw/cot     # change accordingly
source $ENVDIR/bin/activate
module load gcc cuda
module load arrow

export TRITON_CACHE_DIR="/scratch/shenranw/triton_cache"
export HF_HOME="/scratch/shenranw/transformers_cache"

cd /project/6080355/shenranw/ICL

export MODEL="/scratch/shenranw/models/openai-community/gpt2"
export OUT_DIR="out/gpt2"
python extract_activations.py --model $MODEL --add_newlines --task class_to_class --k 4 --split demo --out_dir $OUT_DIR --operator TransformerOperator --stream cache
python extract_activations.py --model $MODEL --add_newlines --task class_to_class_random --k 4 --split demo --out_dir $OUT_DIR --operator TransformerOperator --stream cache

export MODEL="/scratch/shenranw/models/meta-llama/Llama-3.2-1B"
export OUT_DIR="out/Llama-3.2-1B"
python extract_activations.py --model $MODEL --add_newlines --task class_to_class --k 4 --split demo --out_dir $OUT_DIR --operator TransformerOperator --stream cache
python extract_activations.py --model $MODEL --add_newlines --task class_to_class_random --k 4 --split demo --out_dir $OUT_DIR --operator TransformerOperator --stream cache

export MODEL="/scratch/shenranw/models/Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5B"
python extract_activations.py --model $MODEL --add_newlines --task class_to_class --k 4 --split demo --out_dir $OUT_DIR --operator TransformerOperator --stream cache
python extract_activations.py --model $MODEL --add_newlines --task class_to_class_random --k 4 --split demo --out_dir $OUT_DIR --operator TransformerOperator --stream cache

export MODEL="/scratch/shenranw/models/nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python extract_activations.py --model $MODEL --add_newlines --task class_to_class --k 4 --split demo --out_dir $OUT_DIR --operator HymbaOperator --stream cache
python extract_activations.py --model $MODEL --add_newlines --task class_to_class_random --k 4 --split demo --out_dir $OUT_DIR --operator HymbaOperator --stream cache

export MODEL="/scratch/shenranw/models/RWKV/rwkv-6-world-1b6"
export OUT_DIR="out/rwkv-6-world-1b6"
python extract_activations.py --model $MODEL --add_newlines --task class_to_class --k 4 --split demo --out_dir $OUT_DIR --operator RWKVOperator --stream cache
python extract_activations.py --model $MODEL --add_newlines --task class_to_class_random --k 4 --split demo --out_dir $OUT_DIR --operator RWKVOperator --stream cache