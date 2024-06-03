source .bashrc
conda init
conda activate /scratch/shenranw/llm/vllm_env
python3 "./matcha/train.py" experiment=multilingual # ckpt_path="./weights/checkpoint.ckpt"