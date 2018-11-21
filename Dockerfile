FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
LABEL maintainer caffe-maint@googlegroups.com

COPY sources.list  /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        nano \
        fish \
        sudo \
        curl \
        libhdf5-dev \
        libopenblas-dev \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list 
RUN apt update && apt -y install libnccl2=2.2.12-1+cuda8.0 libnccl-dev=2.2.12-1+cuda8.0

RUN pip install --upgrade pip
RUN pip install opencv-python scikit-image protobuf

RUN git clone https://github.com/sumsuddin/caffe.git
WORKDIR /workspace/caffe
RUN git checkout ssd
#COPY Makefile.config /workspace/caffe/Makefile.config
#RUN make -j$(nproc)
#RUN make py
#RUN make test -j$(nproc)
WORKDIR /workspace/caffe/build
RUN cmake .. && make all -j$(nproc) && make install

ENV Caffe_DIR /workspace/caffe/build/

WORKDIR /workspace
#RUN git clone https://github.com/sumsuddin/SFD.git
# Move this copy up after the caffe clone
COPY . /workspace/SDF/

RUN apt install -y nano fish sudo git wget curl

RUN useradd -ms /bin/bash -g root -G sudo -p dockeruser dockeruser
RUN echo dockeruser:dockeruser | chpasswd
USER dockeruser

WORKDIR /home/dockeruser