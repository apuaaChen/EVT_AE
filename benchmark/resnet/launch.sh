#!/bin/bash

echo "pytorch baseline"
python resnet.py -mt torch > ./torch_results.txt

echo "torch channel-last"
python resnet.py -mt torch -cl > ./torch_channel_last_results.txt

echo "inductor"
python resnet.py -mt inductor -cl > ./inductor_results.txt

echo "nvfuser"
python resnet.py -mt aot_ts_nvfuser -cl > ./nvfuser_results.txt

echo "evt"
until python resnet.py -mt gtl -cl > ./evt_results.txt
do
    echo "Error encountered, restarting..."
done   
