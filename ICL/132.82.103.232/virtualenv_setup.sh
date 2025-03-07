#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --account=def-lingjzhu

conda create -n LLM     # change accordingly
conda activate LLM

cd ~/SlurmScripts/ICL/132.82.103.232

conda install cmake
pip install torch==2.5.1
pip install -r ./requirements.txt