FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

RUN apt update && apt install -y git python3 python3-pip
RUN pip3 install --no-cache-dir librosa soundfile numpy faster-whisper torch torchaudio
RUN git clone https://github.com/ufal/whisper_streaming.git

WORKDIR /whisper_streaming

COPY ./initial.wav .

CMD [ "python3", "/whisper_streaming/whisper_online_server.py", \
  "--lan", "ja", \
  "--host", "0.0.0.0", \
  "--port", "43001",  \
  "--warmup-file", "initial.wav", \
  "--task", "transcribe", \
  "--backend", "faster-whisper", \
  "--min-chunk-size", "1", \
  "--vad", \
  "--model_cache_dir", "/models", \
  "--model", "medium"]
