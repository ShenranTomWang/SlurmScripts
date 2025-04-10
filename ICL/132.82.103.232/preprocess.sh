conda activate LLM

cd /home/tomwang/ICL/preprocess

python ./_build_gym.py --build --n_proc 40 --do_test
python ./_build_gym.py --build --n_proc 8 --do_test --task_list "FV" --output_dir "../function_vectors_data" --train_k 1000 --valid_k 100