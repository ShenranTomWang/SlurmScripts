#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --account=def-lingjzhu

module load gcc
module load apptainer

export APPTAINER_DIR=/arc/project/shenranw/jupyter

mkdir $APPTAINER_DIR
cd $APPTAINER_DIR

# install environment
# please find your own environment from docker hub
apptainer pull --name jupyter-datascience.sif docker://jupyter/datascience-notebook