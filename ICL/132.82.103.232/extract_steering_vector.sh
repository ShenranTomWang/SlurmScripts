conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export export CUDA_VISIBLE_DEVICES=3
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export MODEL="Zyphra/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
python extract_activations.py --model $MODEL --add_newlines --dataset tweet_eval-hate --k 16 --split train --out_dir $OUT_DIR --operator ZambaOperator --stream steer --device $DEVICE

export MODEL="nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python extract_activations.py --model $MODEL --add_newlines --dataset tweet_eval-hate --k 16 --split train --out_dir $OUT_DIR --operator HymbaOperator --stream steer --device $DEVICE

export MODEL="state-spaces/mamba-1.4b-hf"
export OUT_DIR="out/mamba-1.4b-hf"
python extract_activations.py --model $MODEL --add_newlines --dataset tweet_eval-hate --k 4 --split train --out_dir $OUT_DIR --operator MambaOperator --stream steer --device $DEVICE

export MODEL="Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5B"
python extract_activations.py --model $MODEL --add_newlines --dataset tweet_eval-hate --k 16 --split train --out_dir $OUT_DIR --operator TransformerOperator --stream steer --device $DEVICE
