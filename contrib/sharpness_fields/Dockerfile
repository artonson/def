FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

# additional packages
RUN apt-get clean \
 && apt-get update -qq \
 && apt-get install -y --no-install-recommends \
    bash \
    git \
    wget \
    python3-pip \
    meshlab \
    xvfb 

# change default python
RUN /bin/bash -c "ln -sf /usr/local/bin/python3.6 /usr/local/bin/python"
RUN /bin/bash -c "ln -sf /usr/local/bin/pip3.6 /usr/local/bin/pip"

# install all python packages in venv
RUN /bin/bash -c "pip3 install --default-timeout=100 numpy==1.16.0 \
                                  Pillow==6.2.0 \
                                  scipy==1.2.2 \
				  h5py==2.10.0 \
                                  six==1.12.0 \
                                  torch==1.2.0 \
                                  torchvision==0.4.0 \
                                  tqdm==4.36.1"

