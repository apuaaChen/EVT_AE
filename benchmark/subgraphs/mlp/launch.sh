#!/bin/bash

# Profile the pytorch
echo "pytorch baseline"
python mlp.py -mt torch > ./torch_results.txt

# Profile the triton
echo "triton"
python mlp.py -mt triton > ./triton_results.txt
python mlp.py -mt triton > ./triton_results.txt

# Profile the autotvm
echo "autotvm"
python mlp.py -mt autotvm > ./autotvm_results.txt

# Profile the tvm with ansor
echo "ansor"
python mlp.py -mt ansor > ./ansor_results.txt

# Profile the tvm with ansor
echo "evt"
python mlp.py -mt evt > ./evt_results.txt