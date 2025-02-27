#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --account=def-lingjzhu

export ENVDIR=/scratch/shenranw/cot     # change accordingly
source $ENVDIR/bin/activate
module load gcc cuda
module load arrow

export TRITON_CACHE_DIR="/scratch/shenranw/triton_cache"
export HF_HOME="/scratch/shenranw/transformers_cache"

cd /project/6080355/shenranw/ICL

export MODEL="/scratch/shenranw/models/nvidia/Hymba-1.5B-Base"
export CACHE_DIR="out/Hymba-1.5B-Base"

export OUT_DIR="out/Hymba-1.5B-Base-ssm-only"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task class_to_class --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --cache2kwargs_kwargs '{"keep_kv": false}'
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task class_to_class_random --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --cache2kwargs_kwargs '{"keep_kv": false}'

export OUT_DIR="out/Hymba-1.5B-Base-kv-only"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task class_to_class --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --cache2kwargs_kwargs '{"keep_ssm": false}'
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --add_newlines --task class_to_class_random --k 4 --n_skips 1 --use_demo_cache --demo_cache_dir $CACHE_DIR --cache2kwargs_kwargs '{"keep_ssm": false}'