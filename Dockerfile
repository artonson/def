FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

## Base packages for ubuntu

# clean the libs list
RUN apt-get clean
RUN apt-get update -qq
RUN apt-get install -y \
    git \
    wget \
    bzip2 \
    htop \
    vim \
    nano \
    g++ \
    make \
    build-essential \
    software-properties-common \
    apt-transport-https \
    sudo \
    gosu \
    libgl1-mesa-glx \
    graphviz \
    tmux \
    screen \
    htop

## Download and install miniconda

RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O /tmp/miniconda.sh

RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:$PATH
RUN echo ${PATH}

## Install base conda packages -- installed through requirements.txt
# RUN conda install -y numpy==1.14.0 jupyter==1.0.0

## Install general requirements for the sharp features
RUN pip install --upgrade pip
COPY requirements.txt /opt/
RUN pip install -r /opt/requirements.txt

# Install pytorch_geometric and friends
RUN pip install --verbose --no-cache-dir torch-scatter
RUN pip install --verbose --no-cache-dir torch-sparse
RUN pip install --verbose --no-cache-dir torch-cluster
RUN pip install --verbose --no-cache-dir torch-spline-conv
RUN pip install torch-geometric

## Torch -- installed through requirements.txt
# RUN conda install pytorch torchvision -c pytorch

## Final stage, staring jupyter and change user and group
##COPY ../../Downloads/docker /root/.jupyter/
#
#ARG UID=1000
#ARG GID=1000
#RUN groupadd -g $GID user && useradd -m -s /bin/bash -u $UID -g user -G root user
#RUN usermod -aG sudo user
#RUN echo "user:user" | chpasswd
#WORKDIR /home/user
#
##RUN mkdir /root/.ssh/
##ADD ../../Downloads/docker /root/.ssh/id_rsa
##RUN chmod 600 /root/.ssh/id_rsa
##RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
##RUN git clone git@github.com:Vahe1994/FloorplanVectorization.git --branch dev
##RUN cd FloorplanVectorization && git pull
##RUN python FloorplanVectorization/setup.py install
#
##COPY --chown=user:user ../../Downloads/docker /home/user/.jupyter/
#COPY runuser.sh /opt/run/
#RUN echo "export PATH='/opt/conda/bin:${PATH}'" >> /home/user/.bashrc
#RUN chmod +x /opt/run/runuser.sh
#
## start custom entrypoint
#ENTRYPOINT ["/opt/run/runuser.sh"]
