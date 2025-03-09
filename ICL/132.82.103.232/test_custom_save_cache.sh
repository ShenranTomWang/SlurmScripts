#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=5-00:00:00
#SBATCH --account=def-lingjzhu

conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export DEVICE="cuda:3"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export MODEL="nvidia/Hymba-1.5B-Base"
export CACHE_DIR="out/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task class_to_class --k 4 --n_skips 1 --use_demo_cache --save_demo_cache --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task class_to_class_random --k 4 --n_skips 1 --use_demo_cache --save_demo_cache --device $DEVICE

export MODEL="RWKV/rwkv-6-world-1b6"
export CACHE_DIR="out/rwkv-6-world-1b6"
export OUT_DIR="out/rwkv-6-world-1b6"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator RWKVOperator --add_newlines --task class_to_class --k 4 --n_skips -1 --use_demo_cache --save_demo_cache --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator RWKVOperator --add_newlines --task class_to_class_random --k 4 --n_skips -1 --use_demo_cache --save_demo_cache --device $DEVICE

# export MODEL="Qwen/Qwen2.5-1.5B"
# export CACHE_DIR="out/Qwen2.5-1.5B"
# export OUT_DIR="out/Qwen2.5-1.5B"
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --add_newlines --task class_to_class --k 4 --n_skips -1 --use_demo_cache --save_demo_cache --device $DEVICE
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --add_newlines --task class_to_class_random --k 4 --n_skips -1 --use_demo_cache --save_demo_cache --device $DEVICE

export MODEL="state-spaces/mamba-1.4b-hf"
export CACHE_DIR="out/mamba-1.4b-hf"
export OUT_DIR="out/mamba-1.4b-hf"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --add_newlines --task class_to_class --k 4 --n_skips -1 --use_demo_cache --save_demo_cache --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --add_newlines --task class_to_class_random --k 4 --n_skips -1 --use_demo_cache --save_demo_cache --device $DEVICE

export MODEL="Zyphra/Zamba2-1.2B"
export CACHE_DIR="out/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --add_newlines --task class_to_class --k 4 --n_skips 1 --use_demo_cache --save_demo_cache --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --add_newlines --task class_to_class_random --k 4 --n_skips 1 --use_demo_cache --save_demo_cache --device $DEVICE
