source .bashrc
conda init
conda activate /scratch/shenranw/llm/vllm_env
conda install -y conda-forge::sox
conda install -y conda-forge::libhwloc
conda install -y conda-forge::rust
pip install -r /project/6080355/shenranw/vocos/requirements.txt
pip install -r /project/6080355/shenranw/Matcha-TTS/requirements.txt
pip install -r /project/6080355/shenranw/vocos/requirements-train.txt