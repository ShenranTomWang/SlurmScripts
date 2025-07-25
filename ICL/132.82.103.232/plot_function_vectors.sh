# conda init
# conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export CUDA_VISIBLE_DEVICES=3
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export MODEL="nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task plot_fv_og --load_dir $OUT_DIR
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task plot_classification --load_dir $OUT_DIR

export MODEL="Zyphra/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task plot_fv_og --load_dir $OUT_DIR
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task plot_classification --load_dir $OUT_DIR