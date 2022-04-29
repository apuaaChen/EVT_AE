import torch
import unittest
from sptrain.meta import bdense2sparse_gold
from sptrain.spmmt import spmmt_bf16_ntn, spmmt_bf16_ntt, spmmt_f16_ntn, spmmt_f16_ntt, spmmt_bf16_nnn, spmmt_bf16_nnt, spmmt_f16_nnn, spmmt_f16_nnt
import torch.nn.functional as F

L = 4
batch_size = 4096
feat_in = 1024
feat_out = 2048


dtype = torch.bfloat16
half = torch.float16


class TestSpMM(unittest.TestCase):
    def test_ntn_bf16(self):
        dense_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=dtype, device="cuda")
        lhs_matrix = torch.randn(size=(L, batch_size, feat_out), dtype=dtype, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.bmm(lhs_matrix, uncompressed).transpose(1, 2)
        output_matrix = spmmt_bf16_ntn(nonzeros, lhs_matrix, metadata)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=1.0))

    def test_ntt_bf16(self):
        dense_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=dtype, device="cuda")
        lhs_matrix = torch.randn(size=(L, batch_size, feat_out), dtype=dtype, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.bmm(lhs_matrix, uncompressed)
        output_matrix = spmmt_bf16_ntt(nonzeros, lhs_matrix, metadata)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=1.0))
    
    def test_ntn_half(self):
        dense_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=half, device="cuda")
        lhs_matrix = torch.randn(size=(L, batch_size, feat_out), dtype=half, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.bmm(lhs_matrix, uncompressed).transpose(1, 2)
        output_matrix = spmmt_f16_ntn(nonzeros, lhs_matrix, metadata)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=1.0))

    def test_ntt_half(self):
        dense_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=half, device="cuda")
        lhs_matrix = torch.randn(size=(L, batch_size, feat_out), dtype=half, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.bmm(lhs_matrix, uncompressed)
        output_matrix = spmmt_f16_ntt(nonzeros, lhs_matrix, metadata)
        
        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=1.0))
    
    def test_nnn_bf16(self):
        dense_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=dtype, device="cuda")
        lhs_matrix = torch.randn(size=(L, feat_out, batch_size), dtype=dtype, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.bmm(lhs_matrix.transpose(1, 2), uncompressed).transpose(1, 2)
        output_matrix = spmmt_bf16_nnn(nonzeros, lhs_matrix, metadata)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=1.0))

    def test_nnt_bf16(self):
        dense_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=dtype, device="cuda")
        lhs_matrix = torch.randn(size=(L, feat_out, batch_size), dtype=dtype, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.bmm(lhs_matrix.transpose(1, 2), uncompressed)
        output_matrix = spmmt_bf16_nnt(nonzeros, lhs_matrix, metadata)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=1.0))
    
    def test_nnn_half(self):
        dense_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=half, device="cuda")
        lhs_matrix = torch.randn(size=(L, feat_out, batch_size), dtype=half, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.bmm(lhs_matrix.transpose(1, 2), uncompressed).transpose(1, 2)
        output_matrix = spmmt_f16_nnn(nonzeros, lhs_matrix, metadata)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=1.0))

    def test_nnt_half(self):
        dense_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=half, device="cuda")
        lhs_matrix = torch.randn(size=(L, feat_out, batch_size), dtype=half, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.bmm(lhs_matrix.transpose(1, 2), uncompressed)
        output_matrix = spmmt_f16_nnt(nonzeros, lhs_matrix, metadata)
        
        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=1.0))

if __name__ == '__main__':
    unittest.main()