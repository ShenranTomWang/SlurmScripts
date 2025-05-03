# conda init
# conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export export CUDA_VISIBLE_DEVICES=1
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export MODEL="Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5B"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_random --load_dir $OUT_DIR
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_original_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_original_random --load_dir $OUT_DIR
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_original_incorrect_mapping --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_original_incorrect_mapping --load_dir $OUT_DIR

export MODEL="meta-llama/Llama-3.2-1B"
export OUT_DIR="out/Llama-3.2-1B"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_random --load_dir $OUT_DIR
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_original_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_original_random --load_dir $OUT_DIR
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_original_incorrect_mapping --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_original_incorrect_mapping --load_dir $OUT_DIR