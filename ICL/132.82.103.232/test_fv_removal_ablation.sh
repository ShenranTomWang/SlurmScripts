# conda init
# conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export CUDA_VISIBLE_DEVICES=2
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export k=16

for p in 0 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20
do
    export MODEL="nvidia/Hymba-1.5B-Base"
    export OUT_DIR="out/Hymba-1.5B-Base"
    export LOG_DIR="logs/Hymba-1.5B-Base/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/zero_ablation/$p/log.log zero_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/zero_ablation/$p/log_scan.log zero_ablation --p $p --stream scan
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/zero_ablation/$p/log_attn.log zero_ablation --p $p --stream attn
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/mean_ablation/$p/log.log mean_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/mean_ablation/$p/log_scan.log mean_ablation --p $p --stream scan
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/mean_ablation/$p/log_attn.log mean_ablation --p $p --stream attn
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_mean_ablation/$p/log.log exclusion_mean_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_mean_ablation/$p/log_scan.log exclusion_mean_ablation --p $p --stream scan --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_mean_ablation/$p/log_attn.log exclusion_mean_ablation --p $p --stream attn --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_zero_ablation/$p/log.log exclusion_zero_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_zero_ablation/$p/log_scan.log exclusion_zero_ablation --p $p --stream scan --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_zero_ablation/$p/log_attn.log exclusion_zero_ablation --p $p --stream attn --exclude_p 0.2

    export MODEL="Qwen/Qwen2.5-1.5B"
    export OUT_DIR="out/Qwen2.5-1.5B"
    export LOG_DIR="logs/Qwen2.5-1.5B/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/zero_ablation/$p/log.log zero_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/mean_ablation/$p/log.log mean_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_mean_ablation/$p/log.log exclusion_mean_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_zero_ablation/$p/log.log exclusion_zero_ablation --p $p --exclude_p 0.2

    export MODEL="meta-llama/Llama-3.2-1B"
    export OUT_DIR="out/Llama-3.2-1B"
    export LOG_DIR="logs/Llama-3.2-1B/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/zero_ablation/$p/log.log zero_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/mean_ablation/$p/log.log mean_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_mean_ablation/$p/log.log exclusion_mean_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_zero_ablation/$p/log.log exclusion_zero_ablation --p $p --exclude_p 0.2

    export MODEL="state-spaces/mamba-1.4b-hf"
    export OUT_DIR="out/mamba-1.4b-hf"
    export LOG_DIR="logs/mamba-1.4b-hf/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/zero_ablation/$p/log.log zero_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/mean_ablation/$p/log.log mean_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_mean_ablation/$p/log.log exclusion_mean_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_zero_ablation/$p/log.log exclusion_zero_ablation --p $p --exclude_p 0.2

    export MODEL="AntonV/mamba2-1.3b-hf"
    export OUT_DIR="out/mamba2-1.3b-hf"
    export LOG_DIR="logs/mamba2-1.3b-hf/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/zero_ablation/$p/log.log zero_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/mean_ablation/$p/log.log mean_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_mean_ablation/$p/log.log exclusion_mean_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_zero_ablation/$p/log.log exclusion_zero_ablation --p $p --exclude_p 0.2

    export MODEL="Zyphra/Zamba2-1.2B"
    export OUT_DIR="out/Zamba2-1.2B"
    export LOG_DIR="logs/Zamba2-1.2B/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/zero_ablation/$p/log.log zero_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/zero_ablation/$p/log_scan.log zero_ablation --p $p --stream scan
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/zero_ablation/$p/log_attn.log zero_ablation --p $p --stream attn
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/mean_ablation/$p/log.log mean_ablation --p $p
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/mean_ablation/$p/log_scan.log mean_ablation --p $p --stream scan
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/mean_ablation/$p/log_attn.log mean_ablation --p $p --stream attn
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_mean_ablation/$p/log.log exclusion_mean_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_mean_ablation/$p/log_scan.log exclusion_mean_ablation --p $p --stream scan --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_mean_ablation/$p/log_attn.log exclusion_mean_ablation --p $p --stream attn --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_zero_ablation/$p/log.log exclusion_zero_ablation --p $p --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_zero_ablation/$p/log_scan.log exclusion_zero_ablation --p $p --stream scan --exclude_p 0.2
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors/exclusion_zero_ablation/$p/log_attn.log exclusion_zero_ablation --p $p --stream attn --exclude_p 0.2
done
