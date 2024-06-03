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
apptainer build llm_env.sif docker://pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
conda create -p /scratch/shenranw/llm/vllm_env --clone base
apptainer run -C --nv --home /project/6080355/shenranw -B /project -B /scratch /project/6080355/shenranw/scripts/llm_env.sif bash /project/6080355/shenranw/scripts/setup_vocos_conda.sh.sh