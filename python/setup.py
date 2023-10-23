from setuptools import setup

setup(
    name='gtl',
    version='0.0.1',
    description='Grafted TMP-Loop Based Compiler for DNN Training',
    author='Zhaodong Chen',
    author_email='chenzd15thu@ucsb.edu',
    package_dir={"": "."},
    packages=[
        "gtl", "gtl.helper", "gtl.compiler", 
        "gtl.compiler.passes", "gtl.compiler.autotuner",
        "gtl.compiler.passes.utils",
        "model_zoo", "model_zoo.bert"]
    # install_requires=['torch']
)