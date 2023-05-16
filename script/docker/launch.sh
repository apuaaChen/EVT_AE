docker run --gpus all --name $1 \
 -v /data:/data -v \
 ../../../SEAL-PICASO-ML-Compiler:/workspace/SEAL-PICASO-ML-Compiler -it \
 -p $2:22 mlcompiler