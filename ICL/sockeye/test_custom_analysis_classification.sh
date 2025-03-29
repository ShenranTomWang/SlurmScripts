#!/bin/bash
#SBATCH --job-name=class_to_class
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

export TRITON_CACHE_DIR="/scratch/st-jzhu71-1/shenranw/triton_cache"    # change accordingly
export HF_HOME="/scratch/st-jzhu71-1/shenranw/transformers_cache"   # change accordingly

cd /scratch/st-jzhu71-1/shenranw/ICL    # change accordingly

export MODEL="nvidia/Hymba-1.5B-Base"   # change accordingly
export OUT_DIR="out/Hymba-1.5B-Base"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task analysis_classification --k 4 --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task analysis_classification_random --k 4 --device $DEVICE

export MODEL="Qwen/Qwen2.5-1.5B"    # change accordingly
export OUT_DIR="out/Qwen2.5-1.5BF"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --task analysis_classification --k 4 --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --task analysis_classification_random --k 4 --device $DEVICE

export MODEL="state-spaces/mamba-1.4b-hf"   # change accordingly
export OUT_DIR="out/mamba-1.4b-hf"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task analysis_classification --k 4 --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task analysis_classification_random --k 4 --device $DEVICE

export MODEL="Zyphra/Zamba2-1.2B"   # change accordingly
export OUT_DIR="out/Zamba2-1.2B"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task analysis_classification --k 4 --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task analysis_classification_random --k 4 --device $DEVICE
