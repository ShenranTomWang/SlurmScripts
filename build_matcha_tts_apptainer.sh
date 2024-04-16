#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --account=def-lingjzhu


module load StdEnv/2023 apptainer/1.2.4

# set cache directory
# please change this to your own /sractch
export APPTAINER_CACHEDIR=~/scratch/cache/apptainer

# install environment
# please find your own environment from docker hub
apptainer build matcha_tts.sif docker://shenranw/matcha_tts:v1