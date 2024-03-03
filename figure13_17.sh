#!/bin/bash

# This file contain the script the reproduce the speedup shown in Figure 13-17
# Launch the profiling of bert
pushd "./benchmark/subgraphs/bert"
bash ./launch.sh
popd

# Launch the profiling of mlp
pushd "./benchmark/subgraphs/mlp"
bash ./launch.sh
popd

# Launch the profiling of resnet
pushd "./benchmark/subgraphs/resnet"
bash ./launch.sh
popd

# Launch the profiling of xml-cnn
pushd "./benchmark/subgraphs/xmlcnn"
bash ./launch.sh
popd

# Launch the profiling of gcn
pushd "./benchmark/subgraphs/gcn"
bash ./launch.sh
popd


# Plot the graph
python ./draw_figure13_17.py