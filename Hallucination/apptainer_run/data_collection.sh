export MODEL="/home/shenranw/CoT/models/google/gemma-2-2b-it"
cd /scratch/st-jzhu71-1/shenranw/Hallucination
export START_IDX=0
# export END_IDX=1000

# export DATASET="QAData"

# export PROBE=1
# python ./data_collection.py

# export PROBE=0
# python ./data_collection.py

export DATASET="TruthfulQA"
export END_IDX=-1

export PROBE=1
python ./data_collection.py

export PROBE=0
python ./data_collection.py