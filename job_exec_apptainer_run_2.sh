#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=24G
#SBATCH --time=7-00:00
#SBATCH --account=def-lingjzhu

module load StdEnv/2023 apptainer/1.2.4
# set cache directory
# Please set all your cache directory to scratch. If you don't, cache files will be placed in your /home
export APPTAINER_CACHEDIR=/scratch/shenranw/cache/apptainer
export MATCHA_TTS_CACHEDIR=/scratch/shenranw/cache/vocos
export APPRTAINER_CWD=/project/6080355/shenranw/vocos

# remove --nv if you don't need a GPU
apptainer run -C --nv --home /project/6080355/shenranw/vocos -B /project -B /scratch /scratch/shenranw/llm_env.sif bash /project/6080355/shenranw/scripts/train_vocos_multilingual.sh