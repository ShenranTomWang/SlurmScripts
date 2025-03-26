conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export export CUDA_VISIBLE_DEVICES=3
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export MODEL="Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5B"
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator TransformerOperator --n_skips -1 baseline
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator TransformerOperator --n_skips -1 intervene
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator TransformerOperator --n_skips -1 intervene_diff

export MODEL="Zyphra/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator ZambaOperator --n_skips -1 baseline
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator ZambaOperator --n_skips -1 intervene
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator ZambaOperator --n_skips -1 intervene_diff

export MODEL="nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator HymbaOperator --n_skips -1 baseline
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator HymbaOperator --n_skips -1 intervene
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator HymbaOperator --n_skips -1 intervene_diff

export MODEL="state-spaces/mamba-1.4b-hf"
export OUT_DIR="out/mamba-1.4b-hf"
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator MambaOperator --n_skips -1 baseline
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator MambaOperator --n_skips -1 intervene
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator MambaOperator --n_skips -1 intervene_diff