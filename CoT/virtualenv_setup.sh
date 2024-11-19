#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=1-00:00
#SBATCH --account=def-lingjzhu

module load StdEnv/2023
module load python/3.10
module load rust
module load arrow
ENVDIR=/scratch/shenranw/cot     # change accordingly
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
pip install --no-index torch transformers sentencepiece pandas plotly nbformat tqdm
pip install transformer_lens