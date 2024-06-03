conda init
source .bashrc
conda activate /scratch/shenranw/llm/vllm_env

cd /project/6080355/shenranw/vocos
python3 -v train.py -c configs/vocos-matcha.yaml