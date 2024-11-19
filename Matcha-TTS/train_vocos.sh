#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=24G
#SBATCH --time=7-00:00
#SBATCH --account=def-lingjzhu

module load StdEnv/2023
module load python/3.10
module load rust
ENVDIR=/scratch/shenranw/tmp
# virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate

cd /project/6080355/shenranw/vocos
python train.py -c configs/vocos-multilingual.yaml