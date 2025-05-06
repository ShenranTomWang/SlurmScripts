# conda init
# conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export export CUDA_VISIBLE_DEVICES=2
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export k=16
export p=0.05

export MODEL="nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
export LOG_DIR="logs/Hymba-1.5B-Base/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_removal_ablation/$p/log.log --ablate_top_p_heads $p

export MODEL="Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5B"
export LOG_DIR="logs/Qwen2.5-1.5B/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_removal_ablation/$p/log.log --ablate_top_p_heads $p

export MODEL="meta-llama/Llama-3.2-1B"
export OUT_DIR="out/Llama-3.2-1B"
export LOG_DIR="logs/Llama-3.2-1B/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_removal_ablation/$p/log.log --ablate_top_p_heads $p

export MODEL="google/gemma-3-1b-pt"
export OUT_DIR="out/gemma-3-1b-pt"
export LOG_DIR="logs/gemma-3-1b-pt/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_removal_ablation/$p/log.log --ablate_top_p_heads $p

export MODEL="state-spaces/mamba-1.4b-hf"
export OUT_DIR="out/mamba-1.4b-hf"
export LOG_DIR="logs/mamba-1.4b-hf/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_removal_ablation/$p/log.log --ablate_top_p_heads $p

export MODEL="AntonV/mamba2-1.3b-hf"
export OUT_DIR="out/mamba2-1.3b-hf"
export LOG_DIR="logs/mamba2-1.3b-hf/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_removal_ablation/$p/log.log --ablate_top_p_heads $p

export MODEL="Zyphra/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
export LOG_DIR="logs/Zamba2-1.2B/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_removal_ablation/$p/log_0_correct.log --ablate_top_p_heads $p
