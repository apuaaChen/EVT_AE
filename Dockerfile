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

# Visualize the adjacency matrix in fusion pass
RUN pip install seaborn

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

# build TVM
RUN mkdir tvm
WORKDIR /workspace/tvm
COPY ./thirdparty/tvm .
RUN mkdir build
COPY ./thirdparty/cmake/config.cmake .
RUN cmake ..
RUN make -j8
ENV TVM_HOME /workspace/tvm
ENV PYTHONPATH /workspace/tvm/python:${PYTHONPATH}

COPY ./thirdparty/pytorch.py ./thirdparty/tvm/python/tvm/relay/frontend/

RUN apt install llvm


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
RUN pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html

RUN cp /workspace/SEAL-PICASSO-ML-Compiler/thirdparty/prebuild_dgl/libdgl_sparse_pytorch_2.1.0a0.so /usr/local/lib/python3.10/dist-packages/dgl/dgl_sparse/

# fix the issue that sparse tensors in torch do not have fake mode
COPY ./python/torch_src/meta_utils.py /usr/local/lib/python3.10/dist-packages/torch/_subclasses/meta_utils.py
COPY ./python/torch_src/aot_autograph.py /usr/local/lib/python3.10/dist-packages/torch/_functorch/aot_autograd.py
