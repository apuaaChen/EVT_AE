FROM nvcr.io/nvidia/pytorch:23.03-py3

WORKDIR /workspace

################################################################################
# install dependencies
################################################################################

RUN apt update

# openssh server
RUN apt install -y --no-install-recommends openssh-server
RUN service ssh start

# required for fx.graph visualization
apt install graphviz
pip install pydot