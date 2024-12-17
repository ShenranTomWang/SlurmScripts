FROM ghcr.io/tilmto/hymba:v1
RUN apt-get update
RUN apt-get install -y git python3-pip automake autoconf squashfuse gocryptfs
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy torch transformers sentencepiece pandas plotly nbformat tqdm nnsight transformer_lens tiktoken protobuf ninja einops
RUN pip install --no-cache-dir notebook