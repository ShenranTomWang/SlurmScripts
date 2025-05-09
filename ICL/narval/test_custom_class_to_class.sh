#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --account=def-lingjzhu

export ENVDIR=/scratch/shenranw/cot     # change accordingly
source $ENVDIR/bin/activate
module load gcc cuda
module load arrow

export TRITON_CACHE_DIR="/scratch/shenranw/triton_cache"
export HF_HOME="/scratch/shenranw/transformers_cache"

cd /project/6080355/shenranw/ICL

# export MODEL="/scratch/shenranw/models/openai-community/gpt2"
# export OUT_DIR="out/gpt2"
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --add_newlines --task class_to_class --k 4 --n_skips -1
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --add_newlines --task class_to_class_random --k 4 --n_skips -1

export MODEL="/scratch/shenranw/models/nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task class_to_class --k 4 --n_skips 1
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task class_to_class_random --k 4 --n_skips 1

# export MODEL="/scratch/shenranw/models/meta-llama/Llama-3.2-1B"
# export OUT_DIR="out/Llama-3.2-1B"
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --add_newlines --task class_to_class --k 4
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --add_newlines --task class_to_class_random --k 4

export MODEL="/scratch/shenranw/models/RWKV/rwkv-6-world-1b6"
export OUT_DIR="out/rwkv-6-world-1b6"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator RWKVOperator --add_newlines --task class_to_class --k 4 --n_skips -1 --use_demo_cache
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator RWKVOperator --add_newlines --task class_to_class_random --k 4 --n_skips -1 --use_demo_cache

export MODEL="/scratch/shenranw/models/Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5BF"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --add_newlines --task class_to_class --k 4 --n_skips -1
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --add_newlines --task class_to_class_random --k 4 --n_skips -1

export MODEL="/scratch/shenranw/models/state-spaces/mamba-1.4b-hf"
export OUT_DIR="out/mamba-1.4b-hf"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --add_newlines --task class_to_class --k 4 --n_skips -1 --use_demo_cache
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --add_newlines --task class_to_class_random --k 4 --n_skips -1 --use_demo_cache

export MODEL="/scratch/shenranw/models/Zyphra/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --add_newlines --task class_to_class --k 4 --n_skips 1 --use_demo_cache
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --add_newlines --task class_to_class_random --k 4 --n_skips 1 --use_demo_cache
