# Build the docker image for the project
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.08-py3
FROM ${FROM_IMAGE_NAME}
RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract

# install openssh server
RUN apt update && apt install -y --no-install-recommends openssh-server
RUN service ssh start

# install functorch and dependencies
RUN pip install git+https://github.com/pytorch/functorch.git@f4a3d5af55897914d516b1329abf2e7b9e95082d
RUN apt install -y graphviz

# install tvm
RUN pip install apache-tvm==0.10.0rc0.dev21

ENV BERT_PREP_WORKING_DIR /workspace/bert/data

WORKDIR /workspace

WORKDIR /workspace/bert
RUN pip install --no-cache-dir \
 tqdm boto3 requests six ipdb h5py nltk progressbar onnxruntime tokenizers>=0.7\
 git+https://github.com/NVIDIA/dllogger wget

RUN apt-get install -y iputils-ping

COPY ./thirdparty/DeepLearningExample/PyTorch/LanguageModeling/BERT .

# Install lddl
RUN conda install -y jemalloc
RUN pip install git+https://github.com/NVIDIA/lddl.git
RUN python -m nltk.downloader punkt

RUN pip install lamb_amp_opt/
