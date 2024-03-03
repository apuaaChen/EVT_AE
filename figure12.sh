#!/bin/bash

# This file contain the script the reproduce the speedup shown in Figure 12

# Launch the profiling of bert
pushd "./benchmark/bert"
bash ./launch.sh
popd

# Launch the profiling of vit
pushd "./benchmark/vit"
bash ./launch.sh
popd

# Launch the profiling of resnet
pushd "./benchmark/resnet"
bash ./launch.sh
popd

# Launch the profiling of xml-cnn
pushd "./benchmark/xmlcnn"
bash ./launch.sh
popd

# Launch the profiling of gcn
pushd "./benchmark/gcn"
bash ./launch.sh
popd


# Plot the graph
python ./draw_figure12.py