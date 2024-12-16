FROM ghcr.io/tilmto/hymba:v1
USER root
RUN apt-get update
RUN apt-get install -y git python3-pip automake autoconf
RUN pip install numpy torch notebook jupyterlab transformers sentencepiece pandas plotly nbformat tqdm nnsight transformer_lens tiktoken protobuf ninja einops