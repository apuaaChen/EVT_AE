#!/bin/bash

# Profile the pytorch
echo "pytorch baseline"
python resnet.py -mt torch > ./torch_results.txt

# Profile the triton
echo "triton"
python resnet.py -mt triton > ./triton_results.txt
python resnet.py -mt triton > ./triton_results.txt

# Profile the autotvm
echo "autotvm"
python resnet.py -mt autotvm > ./autotvm_results.txt

# Profile the tvm with ansor
echo "ansor"
python resnet.py -mt ansor > ./ansor_results.txt

# Profile the tvm with ansor
echo "evt"
python resnet.py -mt evt > ./evt_results.txt