# see: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# 23.02: CUDA 12.0.1, NCCL 2.16.5, CuDNN 8.7.0, python 3.8, pytorch 1.14
# 23.09: CUDA 12.2.1, NCCL 2.18.5, CuDNN 8.9.5, python 3.10, pytorch 2.1.0
# FROM nvcr.io/nvidia/pytorch:23.09-py3
# FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
# FROM image-docker.zuoyebang.cc/asr/train-data2vec-build-offline:test-v.5eb5332ac4b76b6aad7a2f9906ee08c8aca4543f
FROM image-docker.zuoyebang.cc/public/nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# update mirror
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# update timezone
RUN apt-get update --allow-unauthenticated && \
    DEBIAN_FRONTEND=noninteractive TZ=PRC apt-get -y install tzdata && \
    date

RUN apt-get update --allow-unauthenticated && \
    apt-get install -y --allow-unauthenticated libsox-dev  && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    bash-completion vim screen lsof sysstat net-tools strace htop tree nload fio iperf iperf3 iptraf-ng git sudo curl wget

WORKDIR /home/homework


# ##############################################################################
# # Mellanox OFED
# # see: https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/
# https://content.mellanox.com/ofed/MLNX_OFED-5.8-3.0.7.0/MLNX_OFED_LINUX-5.8-3.0.7.0-ubuntu22.04-x86_64.tgz
# https://content.mellanox.com/ofed/MLNX_OFED-5.4-3.1.0.0/MLNX_OFED_LINUX-5.4-3.1.0.0-ubuntu22.04-x86_64.tgz
# ##############################################################################
ENV MLNX_OFED_VERSION=5.4-3.6.8.1
ENV UBUNTU_VERSION=22.04
RUN cd ${WORKDIR} && \
    wget -c "https://content.mellanox.com/ofed/MLNX_OFED-${MLNX_OFED_VERSION}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu${UBUNTU_VERSION}-x86_64.tgz" -O /tmp/ofed.tgz && tar xzf /tmp/ofed.tgz && \
    cd MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu${UBUNTU_VERSION}-x86_64 && \
    ./mlnxofedinstall --user-space-only --without-fw-update --force


# install conda
ENV CONDA_DIR=/opt/miniconda
ENV PATH=${PATH}:${CONDA_DIR}/bin
RUN curl -sSL -o ./miniconda.sh "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-$(uname -m)".sh && \
    echo "Installing conda..." && \
    sha256sum ./miniconda.sh && /bin/bash ./miniconda.sh -bfp $CONDA_DIR && \
    # set timeout for conda installation
    conda config --set remote_read_timeout_secs 900.0 && \
    echo "Installed conda version: $(conda --version), path: $(which conda)"

# RUN conda config --set remote_connect_timeout_secs 30 && \
#     conda config --set remote_read_timeout_secs 120 && \
#     conda config --remove-key channels || true && \
#     conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
#     conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free && \
#     conda config --add channels conda-forge && \
#     conda config --set show_channel_urls yes && \
#     conda config --set channel_priority strict

# 移除 channels 并禁止默认源
RUN conda config --remove-key channels || true && \
    conda config --remove-key default_channels || true && \
    conda config --add default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free && \
    conda config --add channels conda-forge && \
    conda config --set show_channel_urls yes


RUN conda install -y python=3.10

# set pip mirror
RUN echo "Installed python version: $(python --version), path: $(which python), pip path: $(which pip)" && \
    # enable PEP 660 support
    pip config set global.timeout 900 && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --upgrade pip

COPY ./requirements1.txt ./s0/

RUN cd s0 && pip3 install -r requirements1.txt -i https://mirrors.cloud.tencent.com/pypi/simple
