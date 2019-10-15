FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04 

# additional packages
RUN apt-get clean \
 && apt-get update -qq \
 && apt-get install -y --no-install-recommends \
    bash \
    apt-utils \
    sudo \
    git \
    wget \
    vim \
    make \
    python-pip \
    python-tk \
    libgl1-mesa-glx

# install all python packages
RUN /bin/bash -c "pip install --upgrade setuptools && \
		  pip install html5lib \
                              tensorflow-gpu==1.3.0 \
                              tqdm==4.36.1 \
                              h5py==2.10.0 \
                              matplotlib==1.5.3 \
                              numpy==1.14.3 \
                              scipy==1.2.2 \
                              Pillow==6.2.0"

# install graph library
RUN /bin/bash -c "cd /home && git clone https://github.com/wting/python-graph.git && cd python-graph && make install-core"

# load ec-net
RUN /bin/bash -c "cd /home && git clone https://github.com/yulequan/EC-Net.git"
COPY "src/main.py" /home/EC-Net/code/main.py
