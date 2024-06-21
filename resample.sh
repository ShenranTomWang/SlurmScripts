#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --account=def-lingjzhu

module load StdEnv/2023
module load python/3.10
module load rust
ENVDIR=/scratch/shenranw/matcha
# virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate

cd /project/6080355/shenranw/data
python Resample.py
