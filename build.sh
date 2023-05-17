# checkout the cutlass branch
pushd $(pwd)/thirdparty/cutlass
git checkout feature/2.x/epilogue_visitor
popd
docker build -t $1 .