FROM ubuntu:22.04
RUN apt update
RUN apt install -y git python3-pip libsndfile1
RUN apt install -y automake autoconf libtool
RUN git clone https://github.com/rhasspy/espeak-ng && \
    cd espeak-ng && \
    bash autogen.sh && ./configure && make -j8 && make install && \
    ldconfig
RUN git clone git+https://github.com/shivammehta25/Matcha-TTS.git
RUN cd Matcha-TTS
RUN pip install -r requirements.txt
RUN pip install gradio==3.48.0 # Breaks with gradio 4
CMD ["matcha-tts-app"]