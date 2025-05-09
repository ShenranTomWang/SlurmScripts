# conda init
# conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export export CUDA_VISIBLE_DEVICES=2
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

export MODEL="state-spaces/mamba-1.4b-hf"
export OUT_DIR="out/mamba-1.4b-hf"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_random --load_dir $OUT_DIR
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_original_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_original_random --load_dir $OUT_DIR
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_incorrect_mapping --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_incorrect_mapping --load_dir $OUT_DIR

export MODEL="AntonV/mamba2-1.3b-hf"
export OUT_DIR="out/mamba2-1.3b-hf"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_random --load_dir $OUT_DIR
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_original_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_original_random --load_dir $OUT_DIR
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_incorrect_mapping --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_incorrect_mapping --load_dir $OUT_DIR