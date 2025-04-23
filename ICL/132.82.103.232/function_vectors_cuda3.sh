# conda init
# conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export export CUDA_VISIBLE_DEVICES=3
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export MODEL="Zyphra/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_random --load_dir $OUT_DIR
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_original_random --load_dir $OUT_DIR
