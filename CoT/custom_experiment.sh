#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --account=def-lingjzhu
module load StdEnv/2023
module load rust
module load arrow
ENVDIR=/scratch/shenranw/cot
# virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
cd /project/6080355/shenranw/CoT

export MODEL="gemma-2-2b-it"

export STREAM="attn"
export OUTPUT="./experimental_data/${MODEL}/custom_experiment_3/cot/"
export INPUT="Let's think step by step: True or false: Following the \"stay-at-home\" order, the CEO is likely to wear slippers more often than any other shoes even when at work."
export FILENAME="generation_cot.txt"
python ./forward_with_hook_custom.py
export STREAM="attn_scores"
python ./forward_with_hook_custom.py
python ./generate.py

export OUTPUT="./experimental_data/${MODEL}/custom_experiment_3/regular/"
export INPUT="True or false: Following the \"stay-at-home\" order, the CEO is likely to wear slippers more often than any other shoes even when at work."
export FILENAME="generation_reg.txt"
python ./forward_with_hook_custom.py
export STREAM="attn_scores"
python ./forward_with_hook_custom.py
python ./generate.py