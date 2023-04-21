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