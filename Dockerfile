FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    transformers==4.41.2 diffusers==0.28.0 xformers==0.0.25 omegaconf wandb einops pytorch_lightning kornia open-clip-torch==2.24.0

RUN GIT_LFS_SKIP_SMUDGE=1 git clone -b dev https://github.com/camenduru/T2V-Turbo-hf /content/t2v-turbo

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/t2v-turbo/resolve/main/unet_lora.pt -d /content/t2v-turbo/checkpoints -o unet_lora.pt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/t2v-turbo/resolve/main/vc2_model.ckpt -d /content/t2v-turbo/checkpoints -o vc2_model.ckpt

COPY ./worker_runpod.py /content/t2v-turbo/worker_runpod.py
WORKDIR /content/t2v-turbo
CMD python worker_runpod.py