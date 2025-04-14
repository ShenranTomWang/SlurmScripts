# conda init
# conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export export CUDA_VISIBLE_DEVICES=0
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export MODEL="nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_random --device $DEVICE --fv_load_dir $OUT_DIR

export MODEL="Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5B"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_random --device $DEVICE --fv_load_dir $OUT_DIR

export MODEL="state-spaces/mamba-1.4b-hf"
export OUT_DIR="out/mamba-1.4b-hf"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_random --device $DEVICE --fv_load_dir $OUT_DIR

export MODEL="AntonV/mamba2-1.3b-hf"
export OUT_DIR="out/mamba2-1.3b-hf"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_random --device $DEVICE --fv_load_dir $OUT_DIR

export MODEL="Zyphra/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_random --device $DEVICE --fv_load_dir $OUT_DIR
