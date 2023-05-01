################################################################################
# Bert
################################################################################
pushd $(pwd)/../../benchmark/bert
# performance
python bert_benchmarking_20.py -mt torch -cg
python bert_benchmarking_20.py -mt gtl -ps fusion uturn stream -cg
# verification
python bert_benchmarking_20.py -ut
popd
################################################################################
# Vit
################################################################################
pushd $(pwd)/../../benchmark/vit
# performance
python vit_benchmarking_20.py -mt torch -cg
python vit_benchmarking_20.py -mt gtl -ps fusion uturn stream -cg
# verification
python vit_benchmarking_20.py -ut
popd
################################################################################
# ResNet
################################################################################
pushd $(pwd)/../../benchmark/resnet
# performance
python resnet_benchmarking_20.py -mt torch -cg
python resnet_benchmarking_20.py -mt gtl -ps fusion uturn stream -cg
# verification
python resnet_benchmarking_20.py -ut -b 128
popd
################################################################################
# XMLCNN
################################################################################
pushd $(pwd)/../../benchmark/xmlcnn
# performance
python xmlcnn_benchmarking_20.py -mt torch -cg
python xmlcnn_benchmarking_20.py -mt gtl -ps fusion uturn stream -cg
verification
python xmlcnn_benchmarking_20.py -ut
popd
# ################################################################################
# # GCN
# ################################################################################
# pushd $(pwd)/../../benchmark/gcn
# # performance
# python gcn_benchmarking_20.py -mt torch -cg
# python gcn_benchmarking_20.py -mt gtl -ps fusion uturn stream -cg
# # verification
# python gcn_benchmarking_20.py -ut
# popd