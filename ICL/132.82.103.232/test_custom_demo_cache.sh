#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --account=def-lingjzhu

conda activate LLM

export TRITON_CACHE_DIR="/home/shenranw/triton_cache"
export HF_HOME="/home/shenranw/transformers_cache"
export DEVICE="cuda:3"

cd /home/tomwang/ICL

export MODEL="openai-community/gpt2"
export CACHE_DIR="out/gpt2"
export OUT_DIR="out/gpt2"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --add_newlines --task class_to_class --k 4 --n_skips -1 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --add_newlines --task class_to_class_random --k 4 --n_skips -1 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE

export MODEL="nvidia/Hymba-1.5B-Base"
export CACHE_DIR="out/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task class_to_class --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task class_to_class_random --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE

export MODEL="meta-llama/Llama-3.2-1B"
export CACHE_DIR="out/Llama-3.2-1B"
export OUT_DIR="out/Llama-3.2-1B"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --add_newlines --task class_to_class --k 4 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --add_newlines --task class_to_class_random --k 4 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE

export MODEL="RWKV/rwkv-6-world-1b6"
export CACHE_DIR="out/rwkv-6-world-1b6"
export OUT_DIR="out/rwkv-6-world-1b6"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator RWKVOperator --add_newlines --task class_to_class --k 4 --n_skips -1 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator RWKVOperator --add_newlines --task class_to_class_random --k 4 --n_skips -1 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE

export MODEL="Qwen/Qwen2.5-1.5B"
export CACHE_DIR="out/Qwen2.5-1.5BF"
export OUT_DIR="out/Qwen2.5-1.5BF"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --add_newlines --task class_to_class --k 4 --n_skips -1 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --add_newlines --task class_to_class_random --k 4 --n_skips -1 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE