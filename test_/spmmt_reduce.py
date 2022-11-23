import torch
import unittest
from sptrain.meta import bdense2sparse_gold
from sptrain.spmmt_reduce import spmmt_reduce_bf16_ntn, spmmt_reduce_bf16_ntt, spmmt_reduce_f16_ntn, spmmt_reduce_f16_ntt, spmmt_reduce_bf16_nnn, spmmt_reduce_bf16_nnt, spmmt_reduce_f16_nnn, spmmt_reduce_f16_nnt
import torch.nn.functional as F

L = 4
batch_size = 4096
feat_in = 1024
feat_out = 2048


dtype = torch.bfloat16
half = torch.float16

alpha = 0.5


class TestSpMMT_reduce(unittest.TestCase):
    
    def test_nnn_half(self):
        dense_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=half, device="cuda")
        lhs_matrix = torch.randn(size=(L, feat_out, batch_size), dtype=half, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.bmm(lhs_matrix.transpose(1, 2), uncompressed).transpose(1, 2) * alpha
        output_reduce_ref = torch.sum(uncompressed, dim=1) * alpha
        
        output_matrix, output_reduce = spmmt_reduce_f16_nnn(nonzeros, lhs_matrix, metadata, alpha)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=1.0))
        self.assertTrue(torch.allclose(output_reduce_ref, output_reduce, atol=1.0))

if __name__ == '__main__':
    unittest.main()