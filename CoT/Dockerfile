FROM jupyter/datascience-notebook
USER root
RUN apt-get update
RUN apt-get install -y git python3-pip automake autoconf build-essential wget gnupg
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
RUN sudo dpkg -i cuda-keyring_1.1-1_all.deb
RUN sudo apt-get update
RUN apt-get install -y cuda-toolkit-12-6
ENV PATH="/usr/local/cuda/bin:${PATH}" LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN pip install numpy torch transformers sentencepiece pandas plotly nbformat tqdm nnsight transformer_lens tiktoken protobuf ninja einops
RUN pip install triton packaging mamba_ssm flash_attn causal_conv1d