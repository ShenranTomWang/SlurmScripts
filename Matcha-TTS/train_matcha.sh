#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=24G
#SBATCH --time=1-00:00
#SBATCH --account=def-lingjzhu

module load StdEnv/2023
module load python/3.10
module load rust
module load cudacore/.12.2.2
ENVDIR=/scratch/shenranw/matcha
# virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
export USE_MEMORY_EFFICIENT_ATTENTION=0

cd /project/6080355/shenranw/Matcha-TTS
python matcha/train.py experiment=maliseet