# conda init
# conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export CUDA_VISIBLE_DEVICES=0
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export MODEL="nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_random --load_dir $OUT_DIR
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_original_random --load_dir $OUT_DIR
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_incorrect_mapping --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_incorrect_mapping --load_dir $OUT_DIR