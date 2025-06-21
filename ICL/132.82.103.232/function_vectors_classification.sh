# conda init
# conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export CUDA_VISIBLE_DEVICES=1
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export MODEL="nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task classification_random --load_dir $OUT_DIR

export MODEL="Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5B"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task classification_random --load_dir $OUT_DIR

export MODEL="meta-llama/Llama-3.2-1B"
export OUT_DIR="out/Llama-3.2-1B"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task classification_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task classification_random --load_dir $OUT_DIR

export MODEL="state-spaces/mamba-1.4b-hf"
export OUT_DIR="out/mamba-1.4b-hf"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task classification_random --load_dir $OUT_DIR

export MODEL="AntonV/mamba2-1.3b-hf"
export OUT_DIR="out/mamba2-1.3b-hf"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task classification_random --load_dir $OUT_DIR

export MODEL="Zyphra/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task classification_random --load_dir $OUT_DIR