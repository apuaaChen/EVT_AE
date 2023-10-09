FROM nvcr.io/nvidia/pytorch:23.07-py3

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
WORKDIR /workspace/cutlass/python
RUN python setup.py develop --user

# go back to root directory
ENV MLCOMPILER_PATH /workspace/SEAL-PICASSO-ML-Compiler/
WORKDIR /workspace

################################################################################
# benchmark dependencies
################################################################################

# BERT
COPY ./thirdparty/DeepLearningExample/PyTorch/LanguageModeling/BERT /workspace/bert
RUN pip install boto3
# s: substitute; g: make as many substitutions as possible
# This is used to fix the inconsistancy between torch 1.x & 2.0 API
RUN sed -i 's/approximate=True/approximate="tanh"/g' /workspace/bert/modeling.py
# for GCN dataset
RUN pip install ogb
# fix the issue that sparse tensors in torch do not have fake mode
COPY ./python/torch_src/meta_utils.py /usr/local/lib/python3.8/dist-packages/torch/_subclasses/meta_utils.py
