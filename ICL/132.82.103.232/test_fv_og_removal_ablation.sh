# conda init
# conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export CUDA_VISIBLE_DEVICES=3
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export k=16

for p in 0 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20
do
    export MODEL="nvidia/Hymba-1.5B-Base"
    export OUT_DIR="out/Hymba-1.5B-Base"
    export LOG_DIR="logs/Hymba-1.5B-Base/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log.log --use_template zero_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log_scan.log --use_template zero_ablation --p $p --stream scan
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log_attn.log --use_template zero_ablation --p $p --stream attn
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log.log --use_template mean_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log_scan.log --use_template mean_ablation --p $p --stream scan
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log_attn.log --use_template mean_ablation --p $p --stream attn
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log.log --use_template exclusion_mean_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log_scan.log --use_template exclusion_mean_ablation --p $p --stream scan --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log_attn.log --use_template exclusion_mean_ablation --p $p --stream attn --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log.log --use_template exclusion_zero_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log_scan.log --use_template exclusion_zero_ablation --p $p --stream scan --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log_attn.log --use_template exclusion_zero_ablation --p $p --stream attn --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log_mean.log --use_template zero_ablation --p $p --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log_mean_scan.log --use_template zero_ablation --p $p --stream scan --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log_mean_attn.log --use_template zero_ablation --p $p --stream attn --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log_mean.log --use_template mean_ablation --p $p --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log_mean_scan.log --use_template mean_ablation --p $p --stream scan --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log_mean_attn.log --use_template mean_ablation --p $p --stream attn --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log_mean.log --use_template exclusion_mean_ablation --p $p --exclude_p 0.2 --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log_mean_scan.log --use_template exclusion_mean_ablation --p $p --stream scan --exclude_p 0.2 --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log_mean_attn.log --use_template exclusion_mean_ablation --p $p --stream attn --exclude_p 0.2 --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log_mean.log --use_template exclusion_zero_ablation --p $p --exclude_p 0.2 --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log_mean_scan.log --use_template exclusion_zero_ablation --p $p --stream scan --exclude_p 0.2 --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log_mean_attn.log --use_template exclusion_zero_ablation --p $p --stream attn --exclude_p 0.2 --mean_pool

    export MODEL="Qwen/Qwen2.5-1.5B"
    export OUT_DIR="out/Qwen2.5-1.5B"
    export LOG_DIR="logs/Qwen2.5-1.5B/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log.log --use_template zero_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log.log --use_template mean_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log.log --use_template exclusion_mean_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log.log --use_template exclusion_zero_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log_mean.log --use_template zero_ablation --p $p --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log_mean.log --use_template mean_ablation --p $p --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log_mean.log --use_template exclusion_mean_ablation --p $p --exclude_p 0.2 --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log_mean.log --use_template exclusion_zero_ablation --p $p --exclude_p 0.2 --mean_pool

    export MODEL="meta-llama/Llama-3.2-1B"
    export OUT_DIR="out/Llama-3.2-1B"
    export LOG_DIR="logs/Llama-3.2-1B/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log.log --use_template zero_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log.log --use_template mean_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log.log --use_template exclusion_mean_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log.log --use_template exclusion_zero_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log_mean.log --use_template zero_ablation --p $p --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log_mean.log --use_template mean_ablation --p $p --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log_mean.log --use_template exclusion_mean_ablation --p $p --exclude_p 0.2 --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log_mean.log --use_template exclusion_zero_ablation --p $p --exclude_p 0.2 --mean_pool

    export MODEL="state-spaces/mamba-1.4b-hf"
    export OUT_DIR="out/mamba-1.4b-hf"
    export LOG_DIR="logs/mamba-1.4b-hf/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log.log --use_template zero_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log.log --use_template mean_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log.log --use_template exclusion_mean_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log.log --use_template exclusion_zero_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log_mean.log --use_template zero_ablation --p $p --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log_mean.log --use_template mean_ablation --p $p --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log_mean.log --use_template exclusion_mean_ablation --p $p --exclude_p 0.2 --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log_mean.log --use_template exclusion_zero_ablation --p $p --exclude_p 0.2 --mean_pool

    export MODEL="AntonV/mamba2-1.3b-hf"
    export OUT_DIR="out/mamba2-1.3b-hf"
    export LOG_DIR="logs/mamba2-1.3b-hf/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log.log --use_template zero_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log.log --use_template mean_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log.log --use_template exclusion_mean_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log.log --use_template exclusion_zero_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log_mean.log --use_template zero_ablation --p $p --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log_mean.log --use_template mean_ablation --p $p --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log_mean.log --use_template exclusion_mean_ablation --p $p --exclude_p 0.2 --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log_mean.log --use_template exclusion_zero_ablation --p $p --exclude_p 0.2 --mean_pool

    export MODEL="Zyphra/Zamba2-1.2B"
    export OUT_DIR="out/Zamba2-1.2B"
    export LOG_DIR="logs/Zamba2-1.2B/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log.log --use_template zero_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log_scan.log --use_template zero_ablation --p $p --stream scan
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log_attn.log --use_template zero_ablation --p $p --stream attn
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log.log --use_template mean_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log_scan.log --use_template mean_ablation --p $p --stream scan
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log_attn.log --use_template mean_ablation --p $p --stream attn
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log.log --use_template exclusion_mean_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log_scan.log --use_template exclusion_mean_ablation --p $p --stream scan --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log_attn.log --use_template exclusion_mean_ablation --p $p --stream attn --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log.log --use_template exclusion_zero_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log_scan.log --use_template exclusion_zero_ablation --p $p --stream scan --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log_attn.log --use_template exclusion_zero_ablation --p $p --stream attn --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log_mean.log --use_template zero_ablation --p $p --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log_mean_scan.log --use_template zero_ablation --p $p --stream scan --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/zero_ablation/$p/log_mean_attn.log --use_template zero_ablation --p $p --stream attn --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log_mean.log --use_template mean_ablation --p $p --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log_mean_scan.log --use_template mean_ablation --p $p --stream scan --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/mean_ablation/$p/log_mean_attn.log --use_template mean_ablation --p $p --stream attn --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log_mean.log --use_template exclusion_mean_ablation --p $p --exclude_p 0.2 --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log_mean_scan.log --use_template exclusion_mean_ablation --p $p --stream scan --exclude_p 0.2 --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_mean_ablation/$p/log_mean_attn.log --use_template exclusion_mean_ablation --p $p --stream attn --exclude_p 0.2 --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log_mean.log --use_template exclusion_zero_ablation --p $p --exclude_p 0.2 --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log_mean_scan.log --use_template exclusion_zero_ablation --p $p --stream scan --exclude_p 0.2 --mean_pool
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/exclusion_zero_ablation/$p/log_mean_attn.log --use_template exclusion_zero_ablation --p $p --stream attn --exclude_p 0.2 --mean_pool
done
