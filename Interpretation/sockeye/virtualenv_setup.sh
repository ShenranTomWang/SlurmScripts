#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --account=def-jjnunez

module load intel-oneapi-compilers/2023.1.0 python/3.11.6 gcc
export ENVDIR=/scratch/st-jjnunez-1/shenranw/envs/Interpretation     # change accordingly
python -m venv $ENVDIR
source $ENVDIR/bin/activate     # activate the virtual environment

cd /home/shenranw/Interpretation
pip install -r requirements.txt