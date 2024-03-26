#!/bin/bash
JOB_NAME="download_cargo"
#SBATCH --job-name=$JOB_NAME 
#SBATCH --output=$JOB_NAME.out
#SBATCH --error=$JOB_NAME.err
#SBATCH --time=01:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4GB

# Your command goes here
curl https://sh.rustup.rs -sSf | sh

