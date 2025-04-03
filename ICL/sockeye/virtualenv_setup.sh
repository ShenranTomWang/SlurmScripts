#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --account=def-lingjzhu

module load intel-oneapi-compilers/2023.1.0 python/3.11.6
module load cuda
export ENVDIR=/scratch/st-jzhu71-1/shenranw/envs/ICL     # change accordingly
python3.11 -m venv $ENVDIR
source $ENVDIR/bin/activate

pip install torch==2.5.1 triton==3.1.0 wheel sentencepiece pandas nbformat tqdm
pip install transformer_lens tiktoken protobuf ninja einops packaging
pip install notebook mamba-ssm flash-attn causal-conv1d

git clone https://github.com/pytorch-labs/attention-gym.git
cd attention-gym
pip install .
cd ..

pip install -r ./requirements.txt