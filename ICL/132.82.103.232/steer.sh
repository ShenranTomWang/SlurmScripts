conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export export CUDA_VISIBLE_DEVICES=3
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export MODEL="Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5B"
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator TransformerOperator baseline
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator TransformerOperator intervene_direct --keep_scan --keep_attention
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator TransformerOperator intervene_diff --keep_scan --keep_attention

export MODEL="Zyphra/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator ZambaOperator baseline
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator ZambaOperator intervene_direct --keep_scan --keep_attention
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator ZambaOperator intervene_diff --keep_scan --keep_attention

export MODEL="nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator HymbaOperator baseline
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator HymbaOperator intervene_direct --keep_scan --keep_attention
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator HymbaOperator intervene_diff --keep_scan --keep_attention

export MODEL="state-spaces/mamba-1.4b-hf"
export OUT_DIR="out/mamba-1.4b-hf"
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator MambaOperator baseline
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator MambaOperator intervene_direct --keep_scan --keep_attention
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator MambaOperator intervene_diff --keep_scan --keep_attention

export MODEL="AntonV/mamba2-1.3b-hf"
export OUT_DIR="out/mamba2-1.3b-hf"
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator Mamba2Operator baseline
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator Mamba2Operator intervene_direct --keep_scan --keep_attention
python steer.py --model $MODEL --dataset tweet_eval-hate --out_dir $OUT_DIR --load_dir $OUT_DIR --operator Mamba2Operator intervene_diff --keep_scan --keep_attention