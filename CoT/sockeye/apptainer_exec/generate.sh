export HF_HOME="/scratch/st-jzhu71-1/shenranw/transformers_cache"
export INPUT="Let's think step by step: True or false: Anthony can play outside later during the summer, because the days are shorter."
export MODEL="gemma-2-2b-it"
export FILENAME="generation_cot_2.txt"

cd /project/6080355/shenranw/CoT
python ./generate.py