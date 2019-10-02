FROM ubuntu

MAINTAINER gleb bobrovskikh <G.Bobrovskih@skoltech.ru>

RUN apt-get update \
  && apt-get install -y wget \
  && apt-get install -y xz-utils 

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake \
    libopencv-dev \
    libsuitesparse-dev \
    tar \
    libboost-dev libboost-program-options-dev \
    libboost-thread-dev libgmp10-dev \
    libmpfr-dev zlib1g-dev \
    libeigen3-dev libglew1.5-dev libipe-dev \
    libmpfi-dev libqglviewer-dev-qt5 \
    libtbb-dev \
    qtbase5-dev qtscript5-dev libqt5svg5-dev qttools5-dev qttools5-dev-tools

RUN wget https://github.com/CGAL/cgal/releases/download/releases/CGAL-4.14.1/CGAL-4.14.1.tar.xz \
    && tar -xJf CGAL-4.14.1.tar.xz \
    && cd CGAL-4.14.1 && cmake . && make && make install


