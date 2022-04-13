import torch
import unittest
from sptrain.meta import bdense2sparse_gold
from sptrain.spmmt import spmmt_bf16_ntn, spmmt_bf16_ntt, spmmt_f16_ntn, spmmt_f16_ntt
import torch.nn.functional as F


batch_size = 4096
feat_in = 1024
feat_out = 2048


dtype = torch.bfloat16


class TestSpMM(unittest.TestCase):
    def test_ntn_bf16(self):
        dense_matrix = torch.randn(size=(feat_out, feat_in), dtype=dtype, device="cuda")
        lhs_matrix = torch.randn(size=(batch_size, feat_out), dtype=dtype, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.matmul(lhs_matrix, uncompressed).t()
        output_matrix = spmmt_bf16_ntn(nonzeros, lhs_matrix, metadata)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=1.0))

    def test_ntt_bf16(self):
        dense_matrix = torch.randn(size=(feat_out, feat_in), dtype=dtype, device="cuda")
        lhs_matrix = torch.randn(size=(batch_size, feat_out), dtype=dtype, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.matmul(lhs_matrix, uncompressed)
        output_matrix = spmmt_bf16_ntt(nonzeros, lhs_matrix, metadata)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=1.0))
    
    def test_ntn_half(self):
        dense_matrix = torch.randn(size=(feat_out, feat_in), dtype=dtype, device="cuda")
        lhs_matrix = torch.randn(size=(batch_size, feat_out), dtype=dtype, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.matmul(lhs_matrix.to(torch.float16), uncompressed.to(torch.float16)).t()
        output_matrix = spmmt_f16_ntn(nonzeros.to(torch.float16), lhs_matrix.to(torch.float16), metadata)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=1.0))

    def test_ntt_half(self):
        dense_matrix = torch.randn(size=(feat_out, feat_in), dtype=dtype, device="cuda")
        lhs_matrix = torch.randn(size=(batch_size, feat_out), dtype=dtype, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.matmul(lhs_matrix.to(torch.float16), uncompressed.to(torch.float16))
        output_matrix = spmmt_f16_ntt(nonzeros.to(torch.float16), lhs_matrix.to(torch.float16), metadata)
        
        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=1.0))

if __name__ == '__main__':
    unittest.main()