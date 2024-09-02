#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --account=def-lingjzhu

module load StdEnv/2023
module load python/3.10
module load rust
module load cudacore/.12.2.2
ENVDIR=/scratch/shenranw/matcha
# virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
# export USE_MEMORY_EFFICIENT_ATTENTION=1
export BATCHED_SYNTHESIS=1
export MATCHA_CHECKPOINT="./logs/train/monolingual/runs/mikmaw/checkpoints/last.ckpt"
export WANDB_NAME="Mikmaw A100 Vocos epoch=last"
export Y_FILELIST="./data/filelists/mikmaw_test_filelist.txt"
export SPK_FLAG_MONOLINGUAL="AT"
# export LANG_EMB=1
# export SPK_EMB=1

cd /project/6080355/shenranw/Matcha-TTS
python synthesis.py