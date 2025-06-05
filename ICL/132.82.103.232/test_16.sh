# conda init
# conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export CUDA_VISIBLE_DEVICES=3
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export k=16

export MODEL="nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
export LOG_DIR="logs/Hymba-1.5B-Base/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification --k $k --device $DEVICE --log_file $LOG_DIR/classification/log.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_0_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_0_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_25_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_25_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_50_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_50_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_75_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_75_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_random --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_random.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_random_english_words --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_random_english_words.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_random_english_words --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_random_english_words_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task classification_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_incorrect_mapping.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k 0 --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original_random --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_random.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_incorrect_mapping.log

export MODEL="Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5B"
export LOG_DIR="logs/Qwen2.5-1.5B/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification --k $k --device $DEVICE --log_file $LOG_DIR/classification/log.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_0_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_0_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_25_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_25_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_50_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_50_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_75_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_75_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_random --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_random.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_random_english_words --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_random_english_words.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_random_english_words --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_random_english_words_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task classification_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_incorrect_mapping.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_original --k 0 --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_original_random --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_random.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_original_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_original_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_incorrect_mapping.log

export MODEL="meta-llama/Llama-3.2-1B"
export OUT_DIR="out/Llama-3.2-1B"
export LOG_DIR="logs/Llama-3.2-1B/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task classification --k $k --device $DEVICE --log_file $LOG_DIR/classification/log.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task classification_0_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_0_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task classification_25_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_25_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task classification_50_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_50_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task classification_75_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_75_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task classification_random --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_random.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task classification_random_english_words --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_random_english_words.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task classification_random_english_words --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_random_english_words_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task classification_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task classification_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_incorrect_mapping.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_original --k 0 --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_original_random --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_random.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_original_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_original_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_incorrect_mapping.log

export MODEL="google/gemma-3-1b-pt"
export OUT_DIR="out/gemma-3-1b-pt"
export LOG_DIR="logs/gemma-3-1b-pt/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task classification --k $k --device $DEVICE --log_file $LOG_DIR/classification/log.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task classification_0_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_0_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task classification_25_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_25_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task classification_50_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_50_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task classification_75_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_75_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task classification_random --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_random.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task classification_random_english_words --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_random_english_words.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task classification_random_english_words --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_random_english_words_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task classification_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task classification_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_incorrect_mapping.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task function_vectors_original --k 0 --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task function_vectors_original_random --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_random.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task function_vectors_original_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ForwardWrapperTransformerOperator --task function_vectors_original_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_incorrect_mapping.log

export MODEL="state-spaces/mamba-1.4b-hf"
export OUT_DIR="out/mamba-1.4b-hf"
export LOG_DIR="logs/mamba-1.4b-hf/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification --k $k --device $DEVICE --log_file $LOG_DIR/classification/log.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_0_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_0_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_25_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_25_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_50_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_50_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_75_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_75_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_random --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_random.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_random_english_words --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_random_english_words.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_random_english_words --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_random_english_words_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task classification_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_incorrect_mapping.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_original --k 0 --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_original_random --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_random.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_original_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_original_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_incorrect_mapping.log

export MODEL="AntonV/mamba2-1.3b-hf"
export OUT_DIR="out/mamba2-1.3b-hf"
export LOG_DIR="logs/mamba2-1.3b-hf/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification --k $k --device $DEVICE --log_file $LOG_DIR/classification/log.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_0_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_0_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_25_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_25_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_50_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_50_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_75_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_75_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_random --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_random.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_random_english_words --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_random_english_words.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_random_english_words --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_random_english_words_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task classification_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_incorrect_mapping.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_original --k 0 --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_original_random --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_random.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_original_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_original_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_incorrect_mapping.log

export MODEL="Zyphra/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
export LOG_DIR="logs/Zamba2-1.2B/$k"
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification --k $k --device $DEVICE --log_file $LOG_DIR/classification/log.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_0_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_0_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_25_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_25_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_50_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_50_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_75_correct --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_75_correct.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_random --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_random.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_random_english_words --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_random_english_words.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_random_english_words --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_random_english_words_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/classification/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task classification_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/classification/log_incorrect_mapping.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k 0 --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original_random --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_random.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original_incorrect_mapping --k 0 --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_incorrect_mapping_no_demo.log
python test.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original_incorrect_mapping --k $k --device $DEVICE --log_file $LOG_DIR/function_vectors_original/log_incorrect_mapping.log
