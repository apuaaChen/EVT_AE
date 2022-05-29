from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="sptrain_compile",
    version="0.0.1",
    description="Compile passes for dynamic N:M sparsity",
    author="Zhaodong Chen",
    author_email="chenzd15thu@ucsb.edu",
    package_dir={'':"passes"},
    packages=[],
    ext_modules=[
        CppExtension('sptrain_compile',
                     ['passes/register.cpp'], include_dirs=['/opt/conda/lib/python3.8/site-packages/torch']),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)