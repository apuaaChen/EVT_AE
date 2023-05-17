docker run --gpus all --name $1 \
 -v /data:/data -v \
 ${MLCOMPILER_DIR}:/workspace/SEAL-PICASSO-ML-Compiler -it \
 -p $2:22 mlcompiler