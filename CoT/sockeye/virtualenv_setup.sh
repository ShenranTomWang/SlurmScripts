#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --account=def-lingjzhu

module load cuda/12.4.0 intel-oneapi-compilers/2023.1.0 python/3.11.6 gcc
export ENVDIR=/scratch/st-jzhu71-1/shenranw/envs/CoT     # change accordingly
python -m venv $ENVDIR
source $ENVDIR/bin/activate

pip install torch sentencepiece pandas nbformat tqdm
pip install transformer_lens tiktoken protobuf ninja einops triton packaging
pip install notebook mamba-ssm wheel flash-attn causal-conv1d

git clone https://github.com/pytorch-labs/attention-gym.git
cd attention-gym
pip install .
cd ..