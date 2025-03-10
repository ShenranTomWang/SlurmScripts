#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --account=def-lingjzhu

conda create -n LLM     # change accordingly
conda activate LLM

cd /home/tomwang/SlurmScripts/ICL/132.82.103.232

conda install cmake
pip install torch==2.5.1
pip install -r ./requirements.txt
pip install --no-deps datsets==1.4.0
cd /home/tomwang/ICL/preprocess
python _build_gym.py --build --n_proc=40 --do_test
pip install --no-deps datasets==3.2.0