# conda init
# conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export export CUDA_VISIBLE_DEVICES=2
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export MODEL="nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
export LOG_DIR="logs/Hymba-1.5B-Base"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task rethink_demo --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task rethink_demo_0_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_0_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task rethink_demo_25_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_25_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task rethink_demo_50_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_50_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task rethink_demo_75_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_75_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task rethink_demo_random --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_random.log

export MODEL="Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5B"
export LOG_DIR="logs/Qwen2.5-1.5B"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --task rethink_demo --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --task rethink_demo_0_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_0_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --task rethink_demo_25_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_25_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --task rethink_demo_50_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_50_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --task rethink_demo_75_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_75_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator TransformerOperator --task rethink_demo_random --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_random.log

export MODEL="state-spaces/mamba-1.4b-hf"
export OUT_DIR="out/mamba-1.4b-hf"
export LOG_DIR="logs/mamba-1.4b-hf"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task rethink_demo --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task rethink_demo_0_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_0_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task rethink_demo_25_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_25_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task rethink_demo_50_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_50_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task rethink_demo_75_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_75_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task rethink_demo_random --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_random.log

export MODEL="AntonV/mamba2-1.3b-hf"
export OUT_DIR="out/mamba2-1.3b-hf"
export LOG_DIR="logs/mamba2-1.3b-hf"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task rethink_demo --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task rethink_demo_0_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_0_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task rethink_demo_25_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_25_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task rethink_demo_50_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_50_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task rethink_demo_75_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_75_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task rethink_demo_random --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_random.log

export MODEL="Zyphra/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
export LOG_DIR="logs/Zamba2-1.2B"
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task rethink_demo --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task rethink_demo_0_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_0_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task rethink_demo_25_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_25_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task rethink_demo_50_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_50_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task rethink_demo_75_correct --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_75_correct.log
python test_custom.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task rethink_demo_random --k 16 --device $DEVICE --log_file $LOG_DIR/rethink_demo/log_random.log
