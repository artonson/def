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

# Add copyning of our custom code to container and making it
RUN mkdir /home/user/code

COPY ear/src /home/user/code/ear
RUN cd /home/user/code/ear && make

COPY voronoi_1/src /home/user/code/voronoi_1
RUN cd /home/user/code/voronoi_1 && make

# RUN cd /home/user && wget https://raw.githubusercontent.com/artonson/sharp_features/master/CGAL/voronoi_1_scripts/code/voronoi_1.cpp?token=AFP53RFQKG5GBHJD77V3WOC5WANDK -O voronoi_1.cpp && g++ -o voronoi_1 /home/user/voronoi_1.cpp -I/CGAL-4.14.1/include -lCGAL -lgmp
# RUN wget https://raw.githubusercontent.com/artonson/sharp_features/master/CGAL/ear_scripts/code/edge_aware_upsample_point_set.h?token=AFP53RCHKTHP75CNJADFSNS5WAPVK -O edge_aware_upsample_point_set.h && mv edge_aware_upsample_point_set.h /CGAL-4.14.1/include/CGAL/edge_aware_upsample_point_set.h
# RUN cd /home/user && wget https://raw.githubusercontent.com/artonson/sharp_features/master/CGAL/ear_scripts/code/edge_aware_upsampling.cpp?token=AFP53RAYBL77CSBXXMNMZ5C5WAP4C -O edge_aware_upsampling.cpp && g++ -o edge_aware /home/user/edge_aware_upsampling.cpp -I/CGAL-4.14.1/include -lCGAL -lgmp

