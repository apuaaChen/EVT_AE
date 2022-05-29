PYTORCH_DIR=$(python -c 'import os, torch; print(os.path.dirname(os.path.realpath(torch.__file__)))')
PREFIX_DIR=$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')
mkdir -p build && cd build
cmake .. -DPYTORCH_DIR=${PYTORCH_DIR} -DCMAKE_PREFIX_PATH=${PREFIX_DIR}
make -j 24
