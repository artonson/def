FROM nvidia/cuda:10.1-cudnn7-devel
LABEL maintainer="3ddl <adase@skoltech.ru>"

ENV DEBIAN_FRONTEND noninteractive

# Install Ubuntu packages
RUN apt-get clean \
 && apt-get update -qq \
 && apt-get install -y --no-install-recommends \
    libpng-dev \
    libjpeg-dev \
    zlib1g-dev \
    default-jre \
    libopenblas-dev \
    wget \
    sudo \
    gosu \
    libgl1-mesa-glx \
    screen \
    graphviz \
    cmake \
    ninja-build \
    protobuf-compiler \
    libprotobuf-dev \
    bzip2 \
    ca-certificates \
    build-essential \
    software-properties-common \
    curl \
    apt-transport-https \
    openssh-server \
    git-core \
    htop \
    automake \
    libtool \
    pkg-config \
    unzip \
    unrar \
    tree \
    freetds-dev \
    openssh-server \
    vim \
    nano \
    g++ \
    make \
    strace \
    tmux \
    libgl1-mesa-glx \
    rsync \
    net-tools \
    git \
    p7zip-full

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory.
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN echo "export PATH=/home/user/miniconda/bin:$PATH" >> /home/user/.profile
RUN wget --quiet http://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN /bin/bash ~/miniconda.sh -b -p ~/miniconda
RUN rm ~/miniconda.sh
ENV CONDA_AUTO_UPDATE_CONDA=false

# Set path to conda
ENV PATH /home/user/miniconda/bin:$PATH

# Update Miniconda
RUN conda config --add channels conda-forge && \
    conda config --add channels pytorch
RUN conda update -n base -c defaults conda
RUN conda install igl point_cloud_utils libspatialindex pyembree
RUN conda update --all --yes && conda clean --all --yes
RUN conda init bash

# Install Python packages
RUN pip install --no-cache-dir --use-feature=2020-resolver \
    scikit-image tqdm xgboost seaborn scikit-learn pandas \
    xlrd numpy scipy numba matplotlib pillow-simd \
    jupyter catboost statsmodels lightgbm \
    joblib ffmpeg pytest pyyaml \
    termcolor docutils Sphinx recommonmark sphinx_rtd_theme \
    mock yacs tabulate cloudpickle \
    six terminaltables imgaug albumentations \
    shapely psutil portalocker onnx imagesize pydot \
    plyfile rdflib isodate dill dominate psutil \
    tensorboard protobuf future \
    torch==1.5.1 torchvision \
    opencv-python opencv-contrib-python \
    opencv-python-headless Cython lmdb \
    ipyparallel trimesh glob2 h5py pylzma k3d randomcolor nose \
    'git+https://github.com/Toblerity/rtree' \
    tensorflow-hub tensorflow-gan keras tensorflow-gpu scipy
