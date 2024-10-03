#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=24G
#SBATCH --time=1-00:00
#SBATCH --account=def-lingjzhu

module load StdEnv/2023
module load python/3.10
module load rust
ENVDIR=/scratch/shenranw/tmp
# virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate

cd /project/6080355/shenranw/Matcha-TTS
python matcha/utils/generate_data_statistics.py -i multilingual.yaml
