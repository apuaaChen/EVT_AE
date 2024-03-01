#!/bin/bash

# Profile the pytorch
echo "pytorch baseline"
python bert.py -mt torch > ./torch_results.txt

# Profile the inductor
echo "inductor"
python bert.py -mt inductor > ./inductor_results.txt

# Profile the nvfuser
echo "nvfuser"
python bert.py -mt aot_ts_nvfuser > ./nvfuser_results.txt

# Profile the EVT
echo "evt"
until python bert.py -mt gtl > ./evt_results.txt
do
    rm ./compiled_cache.db
    echo "Error encountered, restarting..."
done