#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00
#SBATCH --account=def-lingjzhu

module load StdEnv/2023 apptainer/1.2.4
# set cache directory
# Please set all your cache directory to scratch. If you don't, cache files will be placed in your /home
export APPTAINER_CACHEDIR=/home/shenranw/scratch/cache/apptainer
export MATCHA_TTS_CACHEDIR=/home/shenranw/scratch/cache/matcha_tts

# remove --nv if you don't need a GPU
apptainer run -C --nv --home /project/6080355/shenranw/home -B /project -B /scratch /project/6080355/shenranw/scripts/matcha_tts.sif bash /project/6080355/shenranw/scripts/get_objiwe_stats.sh