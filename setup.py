from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sptrain',
    version='0.0.1',
    description='Custom library for Dynamic Sparse Self-Attention for pytorch',
    author='Zhaodong Chen',
    author_email='chenzd15thu@ucsb.edu',
    package_dir={'':"src"},
    packages=[],
    # packages=['sptrain'],
    ext_modules=[
        CUDAExtension('sptrain.meta', 
                      ['src/cuda/meta.cpp', 'src/cuda/meta_kernel.cu'],
                      extra_compile_args={'cxx':['-lineinfo'], 'nvcc':['-arch=sm_80', '--ptxas-options=-v', '-lineinfo', '-lcublass']},
                      include_dirs=['/home/chenzd15thu/cutlass/include', '/home/chenzd15thu/cutlass/tools/util/include', '/home/chenzd15thu/cutlass/examples/common']),
        CUDAExtension('sptrain.spmm', 
                      ['src/cuda/spmm.cpp', 'src/cuda/spmm_kernel.cu'],
                      extra_cuda_cflags=['-lineinfo'],
                      extra_compile_args={'cxx':['-lineinfo'], 'nvcc':['-arch=sm_80', '--ptxas-options=-v', '-lineinfo', '-lcublass']},
                      include_dirs=['/home/chenzd15thu/cutlass/include', '/home/chenzd15thu/cutlass/tools/util/include', '/home/chenzd15thu/cutlass/examples/common']),
        CUDAExtension('sptrain.gemm', 
                      ['src/cuda/gemm.cpp', 'src/cuda/gemm_kernel.cu'],
                      extra_cuda_cflags=['-lineinfo'],
                      extra_compile_args={'cxx':['-lineinfo'], 'nvcc':['-arch=sm_80', '--ptxas-options=-v', '-lineinfo', '-lcublass']},
                      include_dirs=['/home/chenzd15thu/cutlass/include', '/home/chenzd15thu/cutlass/tools/util/include', '/home/chenzd15thu/cutlass/examples/common']),
        ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)