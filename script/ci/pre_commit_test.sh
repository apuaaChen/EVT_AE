################################################################################
# Bert
################################################################################
pushd $(pwd)/../../benchmark/bert
# performance
python -W ignore bert_benchmarking_20.py -mt torch -cg
python -W ignore bert_benchmarking_20.py -mt gtl -ps fusion uturn stream -cg
# verification
python -W ignore bert_benchmarking_20.py -ut
popd
################################################################################
# Vit
################################################################################
pushd $(pwd)/../../benchmark/vit
# performance
python -W ignore vit_benchmarking_20.py -mt torch -cg
python -W ignore vit_benchmarking_20.py -mt gtl -ps fusion uturn stream -cg
# verification
python -W ignore vit_benchmarking_20.py -ut
popd
################################################################################
# ResNet
################################################################################
pushd $(pwd)/../../benchmark/resnet
# performance
python -W ignore resnet_benchmarking_20.py -mt torch -cg
python -W ignore resnet_benchmarking_20.py -mt gtl -ps fusion uturn stream -cg
# verification
python -W ignore resnet_benchmarking_20.py -ut -b 128
popd
################################################################################
# XMLCNN
################################################################################
pushd $(pwd)/../../benchmark/xmlcnn
# performance
python -W ignore xmlcnn_benchmarking_20.py -mt torch -cg
python -W ignore xmlcnn_benchmarking_20.py -mt gtl -ps fusion uturn stream -cg
# verification
python -W ignore xmlcnn_benchmarking_20.py -ut
popd
################################################################################
# GCN
################################################################################
pushd $(pwd)/../../benchmark/gcn
# performance
python -W ignore gcn_benchmarking_20.py -mt torch -cg
python -W ignore gcn_benchmarking_20.py -mt gtl -ps fusion uturn stream -cg
# verification
python -W ignore gcn_benchmarking_20.py -ut
popd