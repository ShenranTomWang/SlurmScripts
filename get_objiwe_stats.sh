source .bashrc
conda init
conda activate /scratch/shenranw/llm/vllm_env
python3 matcha/utils/generate_data_statistics.py -i objiwe.yaml
