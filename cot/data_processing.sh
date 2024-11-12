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

export BATCH=2
export DATASET="com2sense"
export MODEL_NAME="gemma-2-2b"

cd /project/6080355/shenranw/CoT
python ./data_processing.py