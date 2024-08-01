# pull base image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=utility,compute,graphics \
    PATH=$PATH:/opt/conda/bin \
    DEBIAN_FRONTEND=noninteractive \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.7

WORKDIR /opt

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        libnvinfer8 \
        libnvinfer-plugin8 \
        libopencv-dev \
        tzdata \
        locales \
        apt-utils &&\
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean

RUN cd /usr/lib/x86_64-linux-gnu && \
    ln -s libnvinfer.so.8 libnvinfer.so.7 && \
    ln -s libnvinfer_plugin.so.8 libnvinfer_plugin.so.7

# install miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -P ./downloads/ && \
    bash ./downloads/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -b -p /opt/conda && \
    rm -rf ./downloads

# install additional libraries
COPY requirements.txt .
RUN pip install -U pip && \
    pip install -r requirements.txt && \
    rm -rf ~/.cache/pip \
    rm -f requirements.txt

RUN locale-gen ja_JP.UTF-8

ENV TZ="Asia/Tokyo" \
    LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:en

RUN useradd -m signate
USER signate
WORKDIR /opt/ml