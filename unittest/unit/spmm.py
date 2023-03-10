import torch
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import pycutlass
from pycutlass import *
pycutlass.get_memory_pool(manager="torch")
pycutlass.compiler.nvcc()
import unittest
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.append("/workspace/sparseTraining/Codegen/compiler")
from passes import pass_gemm_fusion, pass_print_graph
import nvtx


dataset = DglNodePropPredDataset(name = "ogbn-mag")

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
g, labels = dataset[0]
g = g.to("cuda")

features = g.ndata["feat"]["paper"]
labels = labels["paper"]
labels = labels.to("cuda").squeeze()

d_i = torch.pow(g.out_degrees(etype='cites').float() + 1, -0.5)
d_j = torch.pow(g.in_degrees(etype='cites').float() + 1, -0.5)
e = dgl.ops.u_mul_v(g, d_i, d_j)[2].to(torch.float16)

row_idx, col_idx, e_idx = g.adj_sparse(fmt="csr", etype='cites')
csr = torch.sparse_csr_tensor(row_idx, col_idx, e)

in_feats = features.shape[1]
n_classes = ((dataset.num_classes + 7) // 8) * 8

class SpmmTestSm80(unittest.TestCase):
    def test_spmm_row_balance(self):
        class Spmm(torch.nn.Module):
            def __init__(self, csr) -> None:
                super().__init__()
                self.csr = csr
            
            def forward(self, x):
                spmm_out = torch.ops.aten.mm(csr, x)
                add_out = torch.ops.aten.add(spmm_out, x)
                dropout_output, mask = torch.ops.aten.native_dropout(
                    add_out, 0.5, True)
                return dropout_output, mask, add_out
        
        ## module instances
        module = Spmm(csr=csr)
        module_reference = Spmm(csr=csr)

        ## run the compiler pass
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        input = torch.randn((features.shape[0], 64), dtype=torch.float16, device="cuda")
        
        ShapeProp(symbolic_traced).propagate(input)

        pass_print_graph(symbolic_traced, "./spmm.svg")

        pass_gemm_fusion(symbolic_traced, symbolic_traced.graph)

        symbolic_traced.recompile()

        pass_print_graph(symbolic_traced, "./spmm_optimized.svg")

        # Test add
        # ref_out = module_reference(input)
        # out = symbolic_traced(input)

        # print(out)
        # print(ref_out)

        # self.assertTrue(torch.allclose(out, ref_out, atol=1))
        # end test add

        ref_dp_out, ref_mask, ref_spmm_out = module_reference(input)
        dp_out, mask, spmm_out = symbolic_traced(input)
        
        dp_out_ref = ref_spmm_out * mask.to(torch.float16)/0.5
        
        print(dp_out)
        print(dp_out_ref)
        self.assertTrue(torch.allclose(dp_out, dp_out_ref, atol=1))

        print(torch.sum(mask) / mask.numel())
        self.assertTrue( 
            torch.allclose(torch.sum(mask) / mask.numel(), torch.Tensor([0.5]).to("cuda"), atol=0.05)
        )

        # for i in range(40):
        #     with nvtx.annotate("torch"):
        #         module_reference(input)
        
        # for i in range(40):
        #     with nvtx.annotate("fused"):
        #         symbolic_traced(input)

if __name__ == '__main__':
    pycutlass.get_memory_pool(manager="torch")
    pycutlass.compiler.nvcc()
    unittest.main()