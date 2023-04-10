################################################################################
# GCN Benchmarking
################################################################################
# Dependencies
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
sys.path.append("/workspace/bert")
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
from gcn_modeling import GCN
import torch.nn.functional as F
from lamb_amp_opt.fused_lamb import FusedLAMBAMP
from aot_helper import compiler_fn, partition_func
from apex import amp
import pycutlass
from pycutlass import *
pycutlass.get_memory_pool(manager="torch")
pycutlass.compiler.nvcc()