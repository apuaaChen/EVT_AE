################################################################################
# Bert
################################################################################
DROPTABLE_SCRIPT=$(pwd)/droptable.py
pushd $(pwd)/../benchmark/bert
# performance
python $DROPTABLE_SCRIPT
# python bert_benchmarking_20.py -mt gtl -ps fusion uturn stream -cg
popd
################################################################################
# Vit
################################################################################
pushd $(pwd)/../benchmark/vit
# performance
python $DROPTABLE_SCRIPT
# python vit_benchmarking_20.py -mt gtl -ps fusion uturn stream -cg
popd
################################################################################
# ResNet
################################################################################
pushd $(pwd)/../benchmark/resnet
# performance
python $DROPTABLE_SCRIPT
# python resnet_benchmarking_20.py -mt gtl -ps fusion uturn stream -cg
popd
################################################################################
# XMLCNN
################################################################################
pushd $(pwd)/../benchmark/xmlcnn
# performance
python $DROPTABLE_SCRIPT
# python xmlcnn_benchmarking_20.py -mt gtl -ps fusion uturn stream -cg
popd
################################################################################
# GCN
################################################################################
pushd $(pwd)/../benchmark/gcn
# performance
python $DROPTABLE_SCRIPT
# python gcn_benchmarking_20.py -mt gtl -ps fusion uturn stream -cg
popd