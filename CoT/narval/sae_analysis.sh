#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --account=def-lingjzhu
module load StdEnv/2023
module load rust
module load arrow
ENVDIR=/scratch/shenranw/cot
# virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate

export MODEL="gemma-2-2b-it"
export INDEX=0
export DATASET="com2sense"
export CHOP_OFF=2
export STREAM="res"

cd /project/6080355/shenranw/CoT
python ./sae_analysis.py