#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --account=def-lingjzhu

module load gcc
module load apptainer

export APPTAINER_CACHEDIR=/scratch/st-jzhu71-1/shenranw/apptainer_cache
export APPTAINER_DIR=/scratch/st-jzhu71-1/shenranw/envs

mkdir $APPTAINER_DIR
cd $APPTAINER_DIR

# install environment
# please find your own environment from docker hub
apptainer pull --force --name hymba.sif docker://shenranw/cot:v5

mkdir -p /scratch/st-jzhu71-1/shenranw/my_jupyter