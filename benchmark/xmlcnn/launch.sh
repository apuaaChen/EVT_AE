#!/bin/bash

echo "pytorch baseline"
python xmlcnn.py -mt torch > ./torch_results.txt

echo "inductor"
python xmlcnn.py -mt inductor > ./inductor_results.txt

echo "nvfuser"
python xmlcnn.py -mt nvprims_nvfuser > ./nvfuser_results.txt

echo "evt"
rm ./compiled_cache.db
until python xmlcnn.py -mt gtl > ./evt_results.txt
do
    rm ./compiled_cache.db
    echo "Error encountered, restarting..."
done   
