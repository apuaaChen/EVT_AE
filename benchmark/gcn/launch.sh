#!/bin/bash

# Profile the pytorch
echo "pytorch baseline"
python gcn.py -mt hand_tuned > ./torch_results.txt

# Profile the nvfuser
echo "nvfuser"
python gcn.py -mt nvprims_nvfuser > ./nvfuser_results.txt

# Profile the EVT
echo "evt"
until python gcn.py -mt gtl > ./evt_results.txt
do
    echo "Error encountered, restarting..."
done   