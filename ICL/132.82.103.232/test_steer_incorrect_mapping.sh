export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export CUDA_VISIBLE_DEVICES=0
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export k=16

export MODEL="nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
export LOG_DIR="logs/Hymba-1.5B-Base/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log.log

export MODEL="Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5B"
export LOG_DIR="logs/Qwen2.5-1.5B/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log.log

export MODEL="meta-llama/Llama-3.2-1B"
export OUT_DIR="out/Llama-3.2-1B"
export LOG_DIR="logs/Llama-3.2-1B/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task classification_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task classification_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log.log

export MODEL="state-spaces/mamba-1.4b-hf"
export OUT_DIR="out/mamba-1.4b-hf"
export LOG_DIR="logs/mamba-1.4b-hf/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log.log

export MODEL="AntonV/mamba2-1.3b-hf"
export OUT_DIR="out/mamba2-1.3b-hf"
export LOG_DIR="logs/mamba2-1.3b-hf/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log.log

export MODEL="Zyphra/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
export LOG_DIR="logs/Zamba2-1.2B/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/log.log

for p in 0 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20
do
    export MODEL="nvidia/Hymba-1.5B-Base"
    export OUT_DIR="out/Hymba-1.5B-Base"
    export LOG_DIR="logs/Hymba-1.5B-Base/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification --k $k --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/$p/log.log steer --p $p

    export MODEL="Qwen/Qwen2.5-1.5B"
    export OUT_DIR="out/Qwen2.5-1.5B"
    export LOG_DIR="logs/Qwen2.5-1.5B/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification --k $k --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/$p/log.log steer --p $p

    export MODEL="meta-llama/Llama-3.2-1B"
    export OUT_DIR="out/Llama-3.2-1B"
    export LOG_DIR="logs/Llama-3.2-1B/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task classification --k $k --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/$p/log.log steer --p $p

    export MODEL="state-spaces/mamba-1.4b-hf"
    export OUT_DIR="out/mamba-1.4b-hf"
    export LOG_DIR="logs/mamba-1.4b-hf/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification --k $k --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/$p/log.log steer --p $p

    export MODEL="AntonV/mamba2-1.3b-hf"
    export OUT_DIR="out/mamba2-1.3b-hf"
    export LOG_DIR="logs/mamba2-1.3b-hf/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification --k $k --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/$p/log.log steer --p $p

    export MODEL="Zyphra/Zamba2-1.2B"
    export OUT_DIR="out/Zamba2-1.2B"
    export LOG_DIR="logs/Zamba2-1.2B/$k"
    python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification --k $k --device $DEVICE --log_file $LOG_DIR/classification/steer_incorrect_mapping/$p/log.log steer --p $p
done