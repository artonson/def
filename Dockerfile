FROM cgal/testsuite-docker:ubuntu

MAINTAINER gleb bobrovskikh <G.Bobrovskih@skoltech.ru>

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     wget \
     apt-utils \
     xz-utils \
     vim \
     sudo

RUN wget https://github.com/CGAL/cgal/releases/download/releases/CGAL-4.14.1/CGAL-4.14.1.tar.xz \
    && tar -xf /CGAL-4.14.1.tar.xz \
    && cd /CGAL-4.14.1 && cmake . && make && make install && rm /CGAL-4.14.1.tar.xz

RUN apt-get update \
  && apt-get install -y libcgal-dev \
     libcgal-demo

# Create a non-root user and switch to it.
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory.
ENV HOME=/home/user
RUN chmod 777 /home/user

## Download and install miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O ~/miniconda.sh \
  && chmod +x ~/miniconda.sh \
  && ~/miniconda.sh -b -p ~/miniconda \
  && rm ~/miniconda.sh \
  && echo "export PATH=/home/user/miniconda/bin:$PATH" >>/home/user/.profile
ENV PATH /home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false 

RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.5 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

RUN pip install --upgrade pip
RUN pip install --verbose --no-cache-dir h5py

