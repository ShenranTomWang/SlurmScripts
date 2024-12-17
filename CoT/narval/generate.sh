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

export INPUT="Let's think step by step: True or false: Anthony can play outside later during the summer, because the days are shorter."
export MODEL="gemma-2-2b-it"
export FILENAME="generation_cot_2.txt"

cd /project/6080355/shenranw/CoT
python ./generate.py

export INPUT="True or false: Anthony can play outside later during the summer, because the days are shorter."
export MODEL="gemma-2-2b-it"
export FILENAME="generation_reg_2.txt"

cd /project/6080355/shenranw/CoT
python ./generate.py
