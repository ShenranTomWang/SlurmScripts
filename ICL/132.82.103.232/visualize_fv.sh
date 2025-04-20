# conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"

cd /home/tomwang/ICL

export OUT_DIR="out/Hymba-1.5B-Base"
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_random --load_dir $OUT_DIR

export OUT_DIR="out/Qwen2.5-1.5B"
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_random --load_dir $OUT_DIR

export OUT_DIR="out/mamba-1.4b-hf"
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_random --load_dir $OUT_DIR

export OUT_DIR="out/mamba2-1.3b-hf"
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_random --load_dir $OUT_DIR

export OUT_DIR="out/Zamba2-1.2B"
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_random --load_dir $OUT_DIR