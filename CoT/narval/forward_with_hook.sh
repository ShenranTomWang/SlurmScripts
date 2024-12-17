#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
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

export BATCH=0
export START_IDX=0
export MAX_IDX=-1
export DATASET="fantasy_reasoning"

cd /project/6080355/shenranw/CoT
python ./forward_with_hook.py

export BATCH=0
export START_IDX=0
export MAX_IDX=900
export DATASET="com2sense"

python ./forward_with_hook.py

export BATCH=1
export START_IDX=900
export MAX_IDX=-1
export DATASET="com2sense"

python ./forward_with_hook.py