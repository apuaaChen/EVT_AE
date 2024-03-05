#!/bin/bash

# Profile the pytorch
echo "pytorch baseline"
python uturn.py -mt apex > ./torch_results.txt

# Profile the triton
echo "triton"
python uturn.py -mt triton > ./triton_results.txt
python uturn.py -mt triton > ./triton_results.txt

# Profile the tvm with ansor
echo "ansor"
python uturn.py -mt tvm > ./ansor_results.txt

# Profile the tvm with ansor
echo "evt"
python uturn.py -mt evt > ./evt_results.txt