FROM nvcr.io/nvidia/pytorch:23.03-py3

WORKDIR /workspace

################################################################################
# install dependencies
################################################################################

RUN apt update

# openssh server
RUN apt install -y --no-install-recommends openssh-server
RUN service ssh start

# tmux
RUN apt install -y tmux

# required for fx.graph visualization
RUN apt install -y graphviz
RUN pip install pydot

# build dgl from source
RUN mkdir dgl
WORKDIR /workspace/dgl
COPY ./thirdparty/dgl .
ENV DGL_HOME /workspace/dgl
RUN bash script/build_dgl.sh -g
WORKDIR /workspace/dgl/python
RUN python setup.py install
RUN python setup.py build_ext --inplace

# build cutlass python interface
RUN apt-get update
WORKDIR /workspace
RUN mkdir cutlass
WORKDIR /workspace/cutlass
COPY ./thirdparty/cutlass .
ENV CUTLASS_PATH /workspace/cutlass
ENV CUDA_INSTALL_PATH /usr/local/cuda
WORKDIR /workspace/cutlass/tools/library/scripts/pycutlass
RUN bash build.sh

# go back to root directory
WORKDIR /workspace

################################################################################
# benchmark dependencies
################################################################################

# for GCN dataset
RUN pip install ogb
