#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --account=def-lingjzhu

module load gcc arrow virtualenv
module load cuda
export ENVDIR=/scratch/shenranw/ICL     # change accordingly
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate

pip install torch sentencepiece pandas nbformat tqdm
pip install transformer_lens tiktoken protobuf ninja einops triton packaging
pip install notebook mamba-ssm wheel flash-attn causal-conv1d

git clone https://github.com/pytorch-labs/attention-gym.git
cd attention-gym
pip install .
cd ..

pip install fla@git+https://github.com/sustcsonglin/flash-linear-attention@452406addd77e37deb7540861f89c70ce678379e

pip install flash-attn@git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2.post1