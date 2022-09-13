# sparseTraining

## Launch docker

```shell
docker run --gpus all --name zdchen_nv_torch_2 -v /data:/data -it -p 8128:22 nvcr.io/nvidia/pytorch:22.02-py3
```

We use NGC pytorch::22.02-py3 and functorch [3f9ac9d](https://github.com/pytorch/functorch/commit/3f9ac9d8b3264da8d9b657e679c55c8e709e814b) due to the compatibility issue, as discussed [here](https://github.com/pytorch/functorch/issues/663)

## Setup user & ssh

**Add a user name to the container**

```shell
adduser --home /home/username username
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
export PATH=/usr/local/nvm/versions/node/v16.6.1/bin:/opt/conda/lib/python3.8/site-packages/torch_tensorrt/bin:/opt/conda/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin

export LD_LIBRARY_PATH=/opt/conda/lib/python3.8/site-packages/torch/lib:/opt/conda/lib/python3.8/site-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
```

```
source ~/.bashrc
```

## Clone file

**Install functorch**

```
 pip install git+https://github.com/pytorch/functorch.git@3f9ac9d8b3264da8d9b657e679c55c8e709e814b
```

**Install sparseTraining**

```
git clone --recursive https://github.com/apuaaChen/sparseTraining.git
```

**Fix the missing header in pytorch**

Add the following file to `/opt/conda/lib/python3.8/site-packages/torch/include/ATen/cuda/nvrtc_stub` (from [source](https://github.com/pytorch/pytorch/blob/17540c5c80f5c6cd4e0fee42ec47d881e46f47f9/aten/src/ATen/cuda/nvrtc_stub/ATenNVRTC.h))

**Install graphviz**
```
apt install graphviz
```