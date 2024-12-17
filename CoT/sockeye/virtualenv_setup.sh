#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --account=def-lingjzhu

module load cuda/12.4.0 python/3.8.10 gcc py-virtualenv
ENVDIR=/scratch/st-jzhu71-1/shenranw/envs/CoT     # change accordingly
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate

pip install torch sentencepiece pandas nbformat tqdm
pip install transformer_lens tiktoken protobuf ninja einops triton packaging
pip install notebook

wget --header="Authorization: Bearer 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" https://huggingface.co/nvidia/Hymba-1.5B-Instruct/resolve/main/setup.sh
bash setup.sh
rm setup.sh