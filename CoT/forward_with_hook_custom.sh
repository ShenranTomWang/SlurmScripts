#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --account=def-lingjzhu
module load StdEnv/2023
module load rust
module load arrow
ENVDIR=/scratch/shenranw/cot
# virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate

export INPUT="Let's think step by step: True or false: Anthony can play outside later during the summer, because the days are shorter.\n **Are days shorter in summer?**"
export MODEL="gemma-2-2b-it"
export STREAM="res"

cd /project/6080355/shenranw/CoT
python ./forward_with_hook_custom.py
