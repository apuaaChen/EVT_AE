#!/bin/bash

Profile the pytorch
echo "pytorch baseline"
python vit.py -mt torch > ./torch_results.txt

echo "inductor"
python vit.py -mt inductor > ./inductor_results.txt

echo "nvfuser"
python vit.py -mt aot_ts_nvfuser > ./nvfuser_results.txt

echo "evt"
until python vit.py -mt gtl > ./evt_results.txt
do
    echo "Error encountered, restarting..."
done   
