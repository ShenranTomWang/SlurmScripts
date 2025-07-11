export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export CUDA_VISIBLE_DEVICES=2
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export k=16

export MODEL="nvidia/Hymba-1.5B-Base"
python layer_steer.py --model $MODEL --operator HymbaOperator --task classification --device $DEVICE --mean_pool

export MODEL="Qwen/Qwen2.5-1.5B"
python layer_steer.py --model $MODEL --operator QwenOperator --task classification --device $DEVICE --mean_pool

export MODEL="meta-llama/Llama-3.2-1B"
python layer_steer.py --model $MODEL --operator LlamaOperator --task classification --device $DEVICE --mean_pool

export MODEL="state-spaces/mamba-1.4b-hf"
python layer_steer.py --model $MODEL --operator MambaOperator --task classification --device $DEVICE --mean_pool

export MODEL="AntonV/mamba2-1.3b-hf"
python layer_steer.py --model $MODEL --operator Mamba2Operator --task classification --device $DEVICE --mean_pool

export MODEL="Zyphra/Zamba2-1.2B"
python layer_steer.py --model $MODEL --operator ZambaOperator --task classification --device $DEVICE --mean_pool