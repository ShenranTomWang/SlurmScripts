#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=1-00:00
#SBATCH --account=def-lingjzhu

module load StdEnv/2023
module load python/3.10
module load rust
ENVDIR=/scratch/shenranw/matcha     # change accordingly
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
pip install --no-index --upgrade pip
pip install gradio==3.43.2
pip install --no-index -r /project/6080355/shenranw/Matcha-TTS/requirements.txt
# pip install --no-index -r /project/6080355/shenranw/vocos/requirements.txt
# pip install -r /project/6080355/shenranw/vocos/requirements-train.txt