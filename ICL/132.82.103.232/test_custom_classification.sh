# conda init
# conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export export CUDA_VISIBLE_DEVICES=0
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

# export MODEL="nvidia/Hymba-1.5B-Base"
# export OUT_DIR="out/Hymba-1.5B-Base"
# export LOG_DIR="logs/Hymba-1.5B-Base"
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_no_demo.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_0_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_0_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_25_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_25_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_50_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_50_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_75_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_75_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_random --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_random.log

# export MODEL="Qwen/Qwen2.5-1.5B"
# export OUT_DIR="out/Qwen2.5-1.5B"
# export LOG_DIR="logs/Qwen2.5-1.5B"
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_no_demo.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_0_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_0_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_25_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_25_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_50_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_50_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_75_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_75_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_random --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_random.log

# export MODEL="meta-llama/Llama-3.2-1B"
# export OUT_DIR="out/Llama-3.2-1B"
# export LOG_DIR="logs/Llama-3.2-1B"
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Llama3Operator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_no_demo.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Llama3Operator --task classification --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Llama3Operator --task classification_0_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_0_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Llama3Operator --task classification_25_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_25_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Llama3Operator --task classification_50_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_50_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Llama3Operator --task classification_75_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_75_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Llama3Operator --task classification_random --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_random.log

export MODEL="google/gemma-3-1b-pt"
export OUT_DIR="out/gemma-3-1b-pt"
export LOG_DIR="logs/gemma-3-1b-pt"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Llama3Operator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_no_demo.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Llama3Operator --task classification --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Llama3Operator --task classification_0_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_0_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Llama3Operator --task classification_25_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_25_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Llama3Operator --task classification_50_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_50_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Llama3Operator --task classification_75_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_75_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Llama3Operator --task classification_random --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_random.log

# export MODEL="state-spaces/mamba-1.4b-hf"
# export OUT_DIR="out/mamba-1.4b-hf"
# export LOG_DIR="logs/mamba-1.4b-hf"
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_no_demo.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_0_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_0_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_25_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_25_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_50_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_50_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_75_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_75_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_random --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_random.log

# export MODEL="AntonV/mamba2-1.3b-hf"
# export OUT_DIR="out/mamba2-1.3b-hf"
# export LOG_DIR="logs/mamba2-1.3b-hf"
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_no_demo.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_0_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_0_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_25_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_25_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_50_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_50_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_75_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_75_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_random --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_random.log

# export MODEL="Zyphra/Zamba2-1.2B"
# export OUT_DIR="out/Zamba2-1.2B"
# export LOG_DIR="logs/Zamba2-1.2B"
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_no_demo.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_0_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_0_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_25_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_25_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_50_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_50_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_75_correct --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_75_correct.log
# python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_random --k 16 --device $DEVICE --log_file $LOG_DIR/classification/log_random.log
