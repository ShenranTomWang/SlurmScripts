export HF_HOME="/scratch/st-jzhu71-1/shenranw/transformers_cache"
export INPUT="What is 103 times 202?"
export MODEL="hymba-1.5b-instruct"
export FILENAME="generation_hymba_1.txt"

cd /scratch/st-jzhu71-1/shenranw/CoT
python ./generate.py