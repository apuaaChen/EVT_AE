# SEAL-PICASSO ML Compiler

This repo is a collection of ML compiler projects from SEAL and PICASSO lab 
targeting single-GPU & distributed training & inference of Deep Learning Models.

## Performance

### Single GPU Training Performance

| Model      | Pytorch         | Ours            | Speedup |
| --         | --              | --              | --      |
| BERT Large | 47627 token/sec | 67005 token/sec | 1.41x   |
| ViT        | 689 image/sec   | 876 image/sec   | 1.27x   |
| ResNet-50  | 2457 image/sec  | 2889 image/sec  | 1.18x   |
| XML-CNN    | 17489 seq/sec   | 52578 seq/sec   | 3.00x   |
| GCN        | 17 epoch/sec    | 98 epoch/sec    | 5.76x   |

## Compatibility

The following environment have been tested
| Hardware| PyTorch Version | CUDA version |
| --      | --              | --           |
| NVIDIA A100 Tensor Core GPU 40 GB | 2.0.0a0+1767026 | CUDA v12.1.66

## Getting Started

We recommend using the docker image:
```bash
git clone https://github.com/apuaaChen/SEAL-PICASSO-ML-Compiler.git
cd SEAL-PICASSO-ML-Compiler
git submodule update --init --recursive
cd ./thirdparty/cutlass && git checkout feature/2.x/epilogue_visitor && cd ../../
export MLCOMPILER_DIR=</path/to/your/SEAL-PICASSO-ML-Compiler/clone>
bash build.sh <image_name>
bash launch.sh <container_name> <port>
```

Inside the docker container, to install gtl library:
```
cd /workspace/SEAL-PICASSO-ML-Compiler/python && bash install.sh
```

## Project Structure

**benchmark**: set of DNN benchmarks for end-to-end test and profiling
```
benchmark/
  bert/     # BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
  gcn/      # Semi-Supervised Classification with Graph Convolutional Networks
  resnet/   # Deep Residual Learning for Image Recognition
  vit/      # An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
  xmlcnn/   # Deep Learning for Extreme Multi-label Text Classification
``` 
**python**: python src code
```
python/
  gtl/            # Source code for GTL: Accelerating Deep Learning Training with Grafted TMP-Loop based Compiler
    compiler/     # GTL Compiler
      autotuner/  # Design space description & xgboost based autotuner
      passes/     # GTL graph-level passes
    helper/       # helpers for amp, aot, profiling and unittests
  torch_src/      # modified torch source code, used to overwritten original torch source code when building the dockerimage
```
**script**: useful scripts
```
script/
  ci/  # scripts for ci test
```
**src**: cpp & cuda src code
```
src/
  cuda/            # cuda source code
    softmax/       # templates for reduce-apply kernels, including softmax, layernorm, and their backward kernels
      epilogue/    # epilogue interface to connect with epilogue visitor tree
      kernel/      # kernel declarations
      threadblock/ # threadblock level modules
    spmm/          # templates for SpMM kernel with CSR input
      epilogue/    # epilogue interface to connect with epilogue visitor tree
      kernel/      # kernel declarations
      threadblock/ # threadblock level modules
```
**thirdparty**: thirdparty dependencies
```
thirdparty/
  AITemplate/
  cutlass/
  DeepLearningExample/
  dgl/
  FlexFlow/
  raf/
  SSGC/
  tvm/
```
**unittest**: unittests
```
unittest/
  fx_passes/    # test the graph-level passes
  unit/         # unittests on basic subgraphs
```

## Benchmarking & Unittest
To run the end-to-end benchmarking & tests:
```bash
cd /workspace/SEAL-PICASSO-ML-Compiler/script/ci/ && bash pre_commit_test.sh
```

## Citations
