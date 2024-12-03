# start from a pytorch prebuilt image that suits the architecture and torch version you need
ARG PYTORCH="2.2.1"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"


# set workdir to /workspace
WORKDIR /workspace

# this ensures that bash will run in the next RUN commands
SHELL ["/bin/bash", "-c"]

RUN conda create --name mem python=3.10
RUN conda init bash
RUN echo "source activate mem" > ~/.bashrc && source ~/.bashrc
COPY ./src /workspace/src
COPY ./prompts /workspace/prompts
COPY ./requirements.txt /workspace/requirements.txt
RUN source activate mem && pip install -r /workspace/requirements.txt

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install basic utilities
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    curl vim nano less git bash-completion screen \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
ENV PYTHONPATH="/workspace/:${PYTHONPATH}"
