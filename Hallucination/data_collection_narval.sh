#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --account=def-lingjzhu
module load StdEnv/2023
module load rust
module load arrow
ENVDIR=/scratch/shenranw/cot
# virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
cd /project/6080355/shenranw/Hallucination

export MODEL="/project/6080355/shenranw/CoT/models/google/gemma-2-2b-it"

# export DATASET="TruthfulQA"
# export START_IDX=0
# export END_IDX=-1

# export PROBE=1
# python ./data_collection.py

# export PROBE=0
# python ./data_collection.py

export START_IDX=0
export END_IDX=1000
export DATASET="QAData"

export PROBE=1
python ./data_collection.py

# export PROBE=0
# python ./data_collection.py