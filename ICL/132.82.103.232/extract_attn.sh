# conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export CUDA_VISIBLE_DEVICES=1
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export MODEL="Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5B"
python extract_activations.py --model $MODEL --task classification --k 16 --split dev --out_dir $OUT_DIR --operator Qwen2Operator --device $DEVICE attn_mean
python extract_activations.py --model $MODEL --task function_vectors_original --k 16 --split dev --out_dir $OUT_DIR --operator Qwen2Operator --device $DEVICE attn_mean

export MODEL="nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python extract_activations.py --model $MODEL --task classification --k 16 --split dev --out_dir $OUT_DIR --operator HymbaOperator --device $DEVICE attn_mean
python extract_activations.py --model $MODEL --task function_vectors_original --k 16 --split dev --out_dir $OUT_DIR --operator HymbaOperator --device $DEVICE attn_mean

export MODEL="state-spaces/mamba-1.4b-hf"
export OUT_DIR="out/mamba-1.4b-hf"
python extract_activations.py --model $MODEL --task classification --k 16 --split dev --out_dir $OUT_DIR --operator MambaOperator --device $DEVICE attn_mean
python extract_activations.py --model $MODEL --task function_vectors_original --k 16 --split dev --out_dir $OUT_DIR --operator MambaOperator --device $DEVICE attn_mean

export MODEL="AntonV/mamba2-1.3b-hf"
export OUT_DIR="out/mamba2-1.3b-hf"
python extract_activations.py --model $MODEL --task classification --k 16 --split dev --out_dir $OUT_DIR --operator Mamba2Operator --device $DEVICE attn_mean
python extract_activations.py --model $MODEL --task function_vectors_original --k 16 --split dev --out_dir $OUT_DIR --operator Mamba2Operator --device $DEVICE attn_mean

export MODEL="Zyphra/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
python extract_activations.py --model $MODEL --task classification --k 16 --split dev --out_dir $OUT_DIR --operator ZambaOperator --device $DEVICE attn_mean
python extract_activations.py --model $MODEL --task function_vectors_original --k 16 --split dev --out_dir $OUT_DIR --operator ZambaOperator --device $DEVICE attn_mean

export MODEL="meta-llama/Llama-3.2-1B"
export OUT_DIR="out/Llama-3.2-1B"
python extract_activations.py --model $MODEL --task classification --k 16 --split dev --out_dir $OUT_DIR --operator LlamaOperator --device $DEVICE attn_mean
python extract_activations.py --model $MODEL --task function_vectors_original --k 16 --split dev --out_dir $OUT_DIR --operator LlamaOperator --device $DEVICE attn_mean
