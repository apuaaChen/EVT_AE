# SEAL-PICASSO ML Compiler

This repo is a collection of ML compiler projects from SEAL and PICASSO lab 
targeting single-GPU & distributed training & inference of Deep Learning Models.

## Performance

## Compatibility

## Getting Started

We recommend using the docker image located in the `./script/docker/` directory.

## Project Structure

## Benchmarking & Unittest

## Contributors

## Citations

****

# Grafted TMP-Loop Based Compiler

## Torch 2.0 update

```bash
# dependency
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Due to the old driver version
```

## Launch docker

```shell
ocker run --gpus all --name ***_nv_torch_4 -v /data:/data -v ~/projects:/workspace -it -p 8125:22 nvcr.io/nvidia/pytorch:22.08-py3
```

We use NGC pytorch::22.08-py3 and functorch 0.3.0a0+f4a3d5a.

## Setup user & ssh

**Add a user name to the container**

```shell
adduser --home /home/*** ***
```

and then follow the instructions.

**Install ssh**

```shell
apt update && apt install -y --no-install-recommends openssh-server
```

**Start ssh**

```shell
service ssh start
```

**Set passwd for root**

```
passwd
```



## Config bashrc

**Add the following two lines**

```
export PATH=/usr/local/nvm/versions/node/v16.19.1/bin:/usr/local/lib/python3.8/dist-packages/torch_tensorrt/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin
export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/torch/lib:/usr/local/lib/python3.8/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-11/lib64
export CUTLASS_PATH=/workspace/cutlass
export CUDA_INSTALL_PATH=/usr/local/cuda
```

```
source ~/.bashrc
```

## Clone file

**Install functorch**

```
pip install git+https://github.com/pytorch/functorch.git@f4a3d5af55897914d516b1329abf2e7b9e95082d
```

**Install sparseTraining**

```
git clone --recursive https://github.com/***/sparseTraining.git
```

**Fix the missing header in pytorch**

Add the following file to `/opt/conda/lib/python3.8/site-packages/torch/include/ATen/cuda/nvrtc_stub` (from [source](https://github.com/pytorch/pytorch/blob/17540c5c80f5c6cd4e0fee42ec47d881e46f47f9/aten/src/ATen/cuda/nvrtc_stub/ATenNVRTC.h))

**Install graphviz**
```
apt install graphviz
pip install pydot
```

**Install TASO**: 
Follow instructions on [INSTALL TASO](https://github.com/jiazhihao/TASO/blob/master/INSTALL.md)

***
### Dependencies
```shell
pip install 'git+https://github.com/NVIDIA/dllogger'
pip install subword-nmt
pip install pytablewriter
pip install einops
pip install deepspeed
```

```shell
cd $CUTLASS_PATH/tools/library/scripts/pycutlass && bash build.sh
```

```
groupadd docker
usermod -a -G docker ***
chgrp -R docker /workspace/bert/
chmod -R g+rwx /workspace/bert/
```

## install GTL
```bash
cd python && python -m pip install ./
```