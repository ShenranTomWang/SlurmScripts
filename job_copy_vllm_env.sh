#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=8G
#SBATCH --time=1-00:00
#SBATCH --account=def-lingjzhu

rm -rf /scratch/shenranw/vllm_env
scp -r shenranw@cedar.alliancecan.ca:/scratch/shenranw/llm/vllm_env /scratch/shenranw