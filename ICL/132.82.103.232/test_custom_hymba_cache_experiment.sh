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

export MODEL="nvidia/Hymba-1.5B-Base"
export CACHE_DIR="out/Hymba-1.5B-Base"

export OUT_DIR="out/Hymba-1.5B-Base-ssm-only"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task analysis_classification --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --cache2kwargs_kwargs '{"keep_kv": false, "keep_conv": false}' --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task analysis_classification_random --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --cache2kwargs_kwargs '{"keep_kv": false, "keep_conv": false}' --device $DEVICE

export OUT_DIR="out/Hymba-1.5B-Base-kv-only"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task analysis_classification --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --cache2kwargs_kwargs '{"keep_ssm": false, "keep_conv": false}' --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task analysis_classification_random --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --cache2kwargs_kwargs '{"keep_ssm": false, "keep_conv": false}' --device $DEVICE

export OUT_DIR="out/Hymba-1.5B-Base-conv-only"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task analysis_classification --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --cache2kwargs_kwargs '{"keep_ssm": false, "keep_kv": false}' --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task analysis_classification_random --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --cache2kwargs_kwargs '{"keep_ssm": false, "keep_kv": false}' --device $DEVICE

for i in {1..32}
do
    export OUT_DIR="out/Hymba-1.5B-Base-layer-$i-only"
    python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task analysis_classification --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --cache2kwargs_kwargs '{"layers": ['$i']}' --device $DEVICE
    python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task analysis_classification_random --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --cache2kwargs_kwargs '{"layers": ['$i']}' --device $DEVICE
done