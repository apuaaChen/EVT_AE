import torch
import unittest
from sptrain.meta import bdense2sparse_gold
from sptrain.spmmt import spmmt_bf16_ntn
import torch.nn.functional as F


batch_size = 4096
feat_in = 1024
feat_out = 2048


dtype = torch.bfloat16


class TestSpMM(unittest.TestCase):
    def test_ntn(self):
        dense_matrix = torch.randn(size=(batch_size, feat_in), dtype=dtype, device="cuda")
        rhs_matrix = torch.randn(size=(feat_out, feat_in), dtype=dtype, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.matmul(uncompressed, rhs_matrix.t())
        output_matrix = spmmt_bf16_ntn(nonzeros, rhs_matrix, metadata)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=0.5))

if __name__ == '__main__':
    unittest.main()