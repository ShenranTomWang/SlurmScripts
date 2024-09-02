source .bashrc
conda init
conda activate /scratch/shenranw/llm/vllm_env
python3 -v "./matcha/train.py" experiment=objiwe # ckpt_path="./weights/checkpoint.ckpt"