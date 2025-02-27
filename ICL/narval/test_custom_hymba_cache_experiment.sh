#!/bin/bash
#SBATCH --job-name=class_to_class
#SBATCH --account=st-jzhu71-1-gpu
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=gpu_mem_32

module load arrow gcc
module load cuda
export ENVDIR=/scratch/shenranw/cot     # change accordingly
source $ENVDIR/bin/activate

export TRITON_CACHE_DIR="/scratch/shenranw/triton_cache"
export HF_HOME="/scratch/shenranw/transformers_cache"

cd /project/6080355/shenranw/ICL

export MODEL="/scratch/shenranw/models/nvidia/Hymba-1.5B-Base"
export CACHE_DIR="out/Hymba-1.5B-Base"

export OUT_DIR="out/Hymba-1.5B-Base-ssm-only"
python test_custom.py --model $MODEL --operator HymbaOperator --add_newlines --task class_to_class --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --cache2kwargs_kwargs '{"keep_kv": false}'
python test_custom.py --model $MODEL --operator HymbaOperator --add_newlines --task class_to_class_random --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --cache2kwargs_kwargs '{"keep_kv": false}'

export OUT_DIR="out/Hymba-1.5B-Base-kv-only"
python test_custom.py --model $MODEL --operator HymbaOperator --add_newlines --task class_to_class --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --cache2kwargs_kwargs '{"keep_ssm": false}'
python test_custom.py --model $MODEL --operator HymbaOperator --add_newlines --task class_to_class_random --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --cache2kwargs_kwargs '{"keep_ssm": false}'