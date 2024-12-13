FROM jupyter/datascience-notebook
USER root
RUN apt update
RUN apt install -y git python3-pip automake autoconf
RUN pip install torch transformers sentencepiece pandas plotly nbformat tqdm nnsight trnasformer_lens tiktoken protobuf ninja einops triton packaging