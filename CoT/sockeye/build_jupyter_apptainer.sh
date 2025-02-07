#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --account=def-lingjzhu

module load gcc
module load apptainer

export APPTAINER_CACHEDIR=/scratch/st-jzhu71-1/shenranw/apptainer_cache
export JUPYTER_APPTAINER_DIR=/scratch/st-jzhu71-1/shenranw/jupyter

mkdir $JUPYTER_APPTAINER_DIR
cd $JUPYTER_APPTAINER_DIR

# install environment
# please find your own environment from docker hub
apptainer pull --force --name jupyter-datascience.sif docker://shenranw/cot:v4

mkdir -p /scratch/st-jzhu71-1/shenranw/my_jupyter