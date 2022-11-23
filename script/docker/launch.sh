docker run --gpus all --name zdchen_nv_torch_5 \
 -v /data/datasets/users/zdchen/workspace/bert:/workspace/bert \
 -v ~/projects/sparseTraining:/workspace/sparseTraining \
 -v ~/projects/cutlass:/workspace/cutlass \
 -v /data:/data \
 -it -p 8225:22 nv_torch