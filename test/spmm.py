import torch
import unittest
from sptrain.meta import bdense2sparse_gold, bdense2sparse
from sptrain.spmm import spmmv2_bf16_nnn, spmmv2_bf16_ntn, spmmv2_bf16_ntt, spmmv2_f16_nnn, spmmv2_f16_ntn, spmmv2_f16_ntt
import torch.nn.functional as F


L = 4
batch_size = 4096
feat_in = 1024
feat_out = 2048


dtype = torch.bfloat16
half = torch.float16


class TestPruning(unittest.TestCase):

    def test_gold_no_abs(self):
        dense_matrix = torch.randn(size=(batch_size, feat_in), dtype=dtype, device="cuda")
        rhs_matrix = torch.eye(n=feat_in, dtype=dtype, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)

        group = int(dense_matrix.numel()/4)
        dense_temp = dense_matrix.detach().reshape(group, 4).to(float)
        index = torch.argsort(dense_temp, dim=1)[:, :int(2)]

        w_b = torch.ones(dense_temp.shape, device=dense_temp.device, dtype=dense_temp.dtype)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(dense_matrix.shape)
        uncompressed_ref = (dense_matrix * w_b).to(torch.bfloat16)

        # Check if the selection is correct
        self.assertTrue(torch.allclose(uncompressed, uncompressed_ref, rtol=1e-9))

        output_matrix_ref = torch.matmul(uncompressed, rhs_matrix)
        output_matrix = spmmv2_bf16_nnn(nonzeros, rhs_matrix, metadata)
        
        # Check if the metadata and nonzeros are correct
        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, rtol=1e-9))
    
    def test_gold_abs(self):
        dense_matrix = torch.randn(size=(batch_size, feat_in), dtype=dtype, device="cuda")
        rhs_matrix = torch.eye(n=feat_in, dtype=dtype, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, True)

        group = int(dense_matrix.numel()/4)
        dense_temp = dense_matrix.abs().detach().reshape(group, 4).to(float)
        index = torch.argsort(dense_temp, dim=1)[:, :int(2)]

        w_b = torch.ones(dense_temp.shape, device=dense_temp.device, dtype=dense_temp.dtype)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(dense_matrix.shape)
        uncompressed_ref = (dense_matrix * w_b).to(torch.bfloat16)

        # Check if the selection is correct
        self.assertTrue(torch.allclose(uncompressed, uncompressed_ref, rtol=1e-9))

        output_matrix_ref = torch.matmul(uncompressed, rhs_matrix)
        output_matrix = spmmv2_bf16_nnn(nonzeros, rhs_matrix, metadata)
        
        # Check if the metadata and nonzeros are correct
        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, rtol=1e-9))
    
    def test_no_abs(self):
        dense_matrix = torch.randn(size=(batch_size, feat_in), dtype=dtype, device="cuda")
        nonzeros_ref, _, metadata_ref = bdense2sparse_gold(dense_matrix, True)
        nonzeros, metadata = bdense2sparse(dense_matrix, True)

        self.assertTrue(torch.allclose(nonzeros_ref, nonzeros, rtol=1e-9))
        self.assertTrue(torch.allclose(metadata_ref, metadata, rtol=1e-9))

    def test_no_abs(self):
        dense_matrix = torch.randn(size=(batch_size, feat_in), dtype=dtype, device="cuda")
        nonzeros_ref, _, metadata_ref = bdense2sparse_gold(dense_matrix, False)
        nonzeros, metadata = bdense2sparse(dense_matrix, False)

        self.assertTrue(torch.allclose(nonzeros_ref, nonzeros, rtol=1e-9))
        self.assertTrue(torch.allclose(metadata_ref, metadata, rtol=1e-9))
    
    def test_gold_no_abs_half(self):
        dense_matrix = torch.randn(size=(batch_size, feat_in), dtype=half, device="cuda")
        rhs_matrix = torch.eye(n=feat_in, dtype=half, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)

        group = int(dense_matrix.numel()/4)
        dense_temp = dense_matrix.detach().reshape(group, 4).to(float)
        index = torch.argsort(dense_temp, dim=1)[:, :int(2)]

        w_b = torch.ones(dense_temp.shape, device=dense_temp.device, dtype=dense_temp.dtype)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(dense_matrix.shape)
        uncompressed_ref = (dense_matrix * w_b).to(torch.float16)

        # Check if the selection is correct
        self.assertTrue(torch.allclose(uncompressed, uncompressed_ref, rtol=1e-9))

        output_matrix_ref = torch.matmul(uncompressed, rhs_matrix)
        output_matrix = spmmv2_f16_nnn(nonzeros, rhs_matrix, metadata)
        
        # Check if the metadata and nonzeros are correct
        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, rtol=1e-9))
    
    def test_gold_abs_half(self):
        dense_matrix = torch.randn(size=(batch_size, feat_in), dtype=half, device="cuda")
        rhs_matrix = torch.eye(n=feat_in, dtype=half, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, True)

        group = int(dense_matrix.numel()/4)
        dense_temp = dense_matrix.abs().detach().reshape(group, 4).to(float)
        index = torch.argsort(dense_temp, dim=1)[:, :int(2)]

        w_b = torch.ones(dense_temp.shape, device=dense_temp.device, dtype=dense_temp.dtype)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(dense_matrix.shape)
        uncompressed_ref = (dense_matrix * w_b).to(torch.float16)

        # Check if the selection is correct
        self.assertTrue(torch.allclose(uncompressed, uncompressed_ref, rtol=1e-9))

        output_matrix_ref = torch.matmul(uncompressed, rhs_matrix)
        output_matrix = spmmv2_f16_nnn(nonzeros, rhs_matrix, metadata)
        
        # Check if the metadata and nonzeros are correct
        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, rtol=1e-9))
    
    def test_abs_half(self):
        dense_matrix = torch.randn(size=(batch_size, feat_in), dtype=half, device="cuda")
        nonzeros_ref, _, metadata_ref = bdense2sparse_gold(dense_matrix, True)
        nonzeros, metadata = bdense2sparse(dense_matrix, True)

        self.assertTrue(torch.allclose(nonzeros_ref, nonzeros, rtol=1e-9))
        self.assertTrue(torch.allclose(metadata_ref, metadata, rtol=1e-9))

    def test_no_abs_half(self):
        dense_matrix = torch.randn(size=(batch_size, feat_in), dtype=half, device="cuda")
        nonzeros_ref, _, metadata_ref = bdense2sparse_gold(dense_matrix, False)
        nonzeros, metadata = bdense2sparse(dense_matrix, False)

        self.assertTrue(torch.allclose(nonzeros_ref, nonzeros, rtol=1e-9))
        self.assertTrue(torch.allclose(metadata_ref, metadata, rtol=1e-9))



class TestSpMM(unittest.TestCase):
    def test_nnn(self):
        dense_matrix = torch.randn(size=(L, batch_size, feat_in), dtype=dtype, device="cuda")
        rhs_matrix = torch.randn(size=(L, feat_in, feat_out), dtype=dtype, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.bmm(uncompressed, rhs_matrix)
        output_matrix = spmmv2_bf16_nnn(nonzeros, rhs_matrix, metadata)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=0.5))

    def test_ntn(self):
        dense_matrix = torch.randn(size=(L, batch_size, feat_in), dtype=dtype, device="cuda")
        rhs_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=dtype, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.bmm(uncompressed, rhs_matrix.transpose(1, 2))
        output_matrix = spmmv2_bf16_ntn(nonzeros, rhs_matrix, metadata)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=0.5))
    
    def test_ntt(self):
        dense_matrix = torch.randn(size=(L, batch_size, feat_in), dtype=dtype, device="cuda")
        rhs_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=dtype, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.bmm(uncompressed, rhs_matrix.transpose(1, 2)).transpose(1, 2)
        output_matrix = spmmv2_bf16_ntt(nonzeros, rhs_matrix, metadata)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=0.5))
    
    def test_nnn_half(self):
        dense_matrix = torch.randn(size=(L, batch_size, feat_in), dtype=half, device="cuda")
        rhs_matrix = torch.randn(size=(L, feat_in, feat_out), dtype=half, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.bmm(uncompressed, rhs_matrix)
        output_matrix = spmmv2_f16_nnn(nonzeros, rhs_matrix, metadata)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=0.5))

    def test_ntn_half(self):
        dense_matrix = torch.randn(size=(L, batch_size, feat_in), dtype=half, device="cuda")
        rhs_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=half, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.bmm(uncompressed, rhs_matrix.transpose(1, 2))
        output_matrix = spmmv2_f16_ntn(nonzeros, rhs_matrix, metadata)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=0.5))
    
    def test_ntt_half(self):
        dense_matrix = torch.randn(size=(L, batch_size, feat_in), dtype=half, device="cuda")
        rhs_matrix = torch.randn(size=(L, feat_out, feat_in), dtype=half, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix, False)
        output_matrix_ref = torch.bmm(uncompressed, rhs_matrix.transpose(1, 2)).transpose(1, 2)
        output_matrix = spmmv2_f16_ntt(nonzeros, rhs_matrix, metadata)

        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, atol=0.5))

if __name__ == '__main__':
    unittest.main()