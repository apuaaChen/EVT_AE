#!/bin/bash

# Profile the pytorch
echo "pytorch baseline"
python selfattention.py -mt torch > ./torch_results.txt

# Profile the triton
echo "triton"
python selfattention.py -mt triton > ./triton_results.txt

# Profile the autotvm
echo "autotvm"
python selfattention.py -mt autotvm > ./autotvm_results.txt

# Profile the tvm with ansor
echo "ansor"
python selfattention.py -mt ansor > ./ansor_results.txt

# Profile the tvm with ansor
echo "evt"
python selfattention.py -mt evt > ./evt_results.txt