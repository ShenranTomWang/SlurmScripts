conda init
source .bashrc
conda activate /scratch/shenranw/llm/vllm_env

cd /project/6080355/shenranw/vocos
python3 train.py -c configs/vocos-multilingual.yaml