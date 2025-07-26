# conda init
# conda activate LLM

export TRITON_CACHE_DIR="/home/tomwang/triton_cache"
export HF_HOME="/home/tomwang/transformers_cache"
export CUDA_VISIBLE_DEVICES=3
export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

cd /home/tomwang/ICL

for p in 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20
do
    export MODEL="nvidia/Hymba-1.5B-Base"
    export LOAD_DIR="out/Hymba-1.5B-Base"
    export LOG_FILE="logs/Hymba-1.5B-Base/top_p_overlap/${p}/log.log"
    python top_p_overlap.py --operator HymbaOperator --model ${MODEL} --task1 classification_random --task2 function_vectors_original_random --log_file ${LOG_FILE} --p ${p} --fv_map_load_dir ${LOAD_DIR}

    export MODEL="Qwen/Qwen2.5-1.5B"
    export LOAD_DIR="out/Qwen2.5-1.5B"
    export LOG_FILE="logs/Qwen2.5-1.5B/top_p_overlap/${p}/log.log"
    python top_p_overlap.py --operator Qwen2Operator --model ${MODEL} --task1 classification_random --task2 function_vectors_original_random --log_file ${LOG_FILE} --p ${p} --fv_map_load_dir ${LOAD_DIR}

    export MODEL="meta-llama/Llama-3.2-1B"
    export LOAD_DIR="out/Llama-3.2-1B"
    export LOG_FILE="logs/Llama-3.2-1B/top_p_overlap/${p}/log.log"
    python top_p_overlap.py --operator LlamaOperator --model ${MODEL} --task1 classification_random --task2 function_vectors_original_random --log_file ${LOG_FILE} --p ${p} --fv_map_load_dir ${LOAD_DIR}

    export MODEL="state-spaces/mamba-1.4b-hf"
    export LOAD_DIR="out/mamba-1.4b-hf"
    export LOG_FILE="logs/mamba-1.4b-hf/top_p_overlap/${p}/log.log"
    python top_p_overlap.py --operator MambaOperator --model ${MODEL} --task1 classification_random --task2 function_vectors_original_random --log_file ${LOG_FILE} --p ${p} --fv_map_load_dir ${LOAD_DIR}

    export MODEL="AntonV/mamba2-1.3b-hf"
    export LOAD_DIR="out/mamba2-1.3b-hf"
    export LOG_FILE="logs/mamba2-1.3b-hf/top_p_overlap/${p}/log.log"
    python top_p_overlap.py --operator Mamba2Operator --model ${MODEL} --task1 classification_random --task2 function_vectors_original_random --log_file ${LOG_FILE} --p ${p} --fv_map_load_dir ${LOAD_DIR}

    export MODEL="Zyphra/Zamba2-1.2B"
    export LOAD_DIR="out/Zamba2-1.2B"
    export LOG_FILE="logs/Zamba2-1.2B/top_p_overlap/${p}/log.log"
    python top_p_overlap.py --operator ZambaOperator --model ${MODEL} --task1 classification_random --task2 function_vectors_original_random --log_file ${LOG_FILE} --p ${p} --fv_map_load_dir ${LOAD_DIR}
done
