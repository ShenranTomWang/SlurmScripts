FROM jupyter/datascience-notebook
USER root
RUN apt update
RUN apt install -y git python3-pip automake autoconf
RUN git clone https://github.com/saprmarks/geometry-of-truth.git
RUN cd geometry-of-truth
RUN ls
RUN pip install -r ./requirements.txt
RUN pip install nnsight