#!/bin/bash

# Profile the pytorch
echo "pytorch baseline"
python xmlcnn.py -mt torch > ./torch_results.txt

# Profile the triton
echo "triton"
python xmlcnn.py -mt triton > ./triton_results.txt
python xmlcnn.py -mt triton > ./triton_results.txt

# Profile the autotvm
echo "autotvm"
python xmlcnn.py -mt autotvm > ./autotvm_results.txt

# Profile the tvm with ansor
echo "evt"
python xmlcnn.py -mt evt > ./evt_results.txt