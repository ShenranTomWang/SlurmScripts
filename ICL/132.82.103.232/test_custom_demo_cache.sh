conda activate LLM

export TRITON_CACHE_DIR="/home/shenranw/triton_cache"
export HF_HOME="/home/shenranw/transformers_cache"
export export CUDA_VISIBLE_DEVICES=3
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export MODEL="openai-community/gpt2"
export CACHE_DIR="out/gpt2"
export OUT_DIR="out/gpt2"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --task analysis_classification --k 4 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --task analysis_classification_random --k 4 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE

export MODEL="nvidia/Hymba-1.5B-Base"
export CACHE_DIR="out/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task analysis_classification --k 4 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task analysis_classification_random --k 4 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE

export MODEL="meta-llama/Llama-3.2-1B"
export CACHE_DIR="out/Llama-3.2-1B"
export OUT_DIR="out/Llama-3.2-1B"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --task analysis_classification --k 4 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --task analysis_classification_random --k 4 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE

export MODEL="RWKV/rwkv-6-world-1b6"
export CACHE_DIR="out/rwkv-6-world-1b6"
export OUT_DIR="out/rwkv-6-world-1b6"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator RWKVOperator --task analysis_classification --k 4 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator RWKVOperator --task analysis_classification_random --k 4 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE

export MODEL="Qwen/Qwen2.5-1.5B"
export CACHE_DIR="out/Qwen2.5-1.5BF"
export OUT_DIR="out/Qwen2.5-1.5BF"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --task analysis_classification --k 4 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --task analysis_classification_random --k 4 --use_demo_cache --demo_cache_dir $CACHE_DIR --device $DEVICE