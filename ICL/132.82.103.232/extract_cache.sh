conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export export CUDA_VISIBLE_DEVICES=3
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export MODEL="Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5B"
python extract_activations.py --model $MODEL --add_newlines --task analysis_classification --k 4 --split demo --out_dir $OUT_DIR --operator TransformerOperator --stream cache --device $DEVICE
python extract_activations.py --model $MODEL --add_newlines --task analysis_classification_random --k 4 --split demo --out_dir $OUT_DIR --operator TransformerOperator --stream cache --device $DEVICE

export MODEL="nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python extract_activations.py --model $MODEL --add_newlines --task analysis_classification --k 4 --split demo --out_dir $OUT_DIR --operator HymbaOperator --stream cache --device $DEVICE
python extract_activations.py --model $MODEL --add_newlines --task analysis_classification_random --k 4 --split demo --out_dir $OUT_DIR --operator HymbaOperator --stream cache --device $DEVICE

export MODEL="RWKV/rwkv-6-world-1b6"
export OUT_DIR="out/rwkv-6-world-1b6"
python extract_activations.py --model $MODEL --add_newlines --task analysis_classification --k 4 --split demo --out_dir $OUT_DIR --operator RWKVOperator --stream cache --device $DEVICE
python extract_activations.py --model $MODEL --add_newlines --task analysis_classification_random --k 4 --split demo --out_dir $OUT_DIR --operator RWKVOperator --stream cache --device $DEVICE

export MODEL="state-spaces/mamba-1.4b-hf"
export OUT_DIR="out/mamba-1.4b-hf"
python extract_activations.py --model $MODEL --add_newlines --task analysis_classification --k 4 --split demo --out_dir $OUT_DIR --operator MambaOperator --stream cache --device $DEVICE
python extract_activations.py --model $MODEL --add_newlines --task analysis_classification_random --k 4 --split demo --out_dir $OUT_DIR --operator MambaOperator --stream cache --device $DEVICE

export MODEL="Zyphra/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
python extract_activations.py --model $MODEL --add_newlines --task analysis_classification --k 4 --split demo --out_dir $OUT_DIR --operator ZambaOperator --stream cache --device $DEVICE
python extract_activations.py --model $MODEL --add_newlines --task analysis_classification_random --k 4 --split demo --out_dir $OUT_DIR --operator ZambaOperator --stream cache --device $DEVICE
