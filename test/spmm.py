import torch
import nvtx
import unittest
from sptrain.meta import bdense2sparse_gold, bdense2sparse
from sptrain.spmm import spmmv2_bf16


batch_size = 512
feat_in = 1024
feat_out = 1024


dtype = torch.bfloat16


class TestPruning(unittest.TestCase):
    def test_gold(self):
        dense_matrix = torch.randn(size=(batch_size, feat_in), dtype=dtype, device="cuda")
        rhs_matrix = torch.eye(n=feat_in, dtype=dtype, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix)
        output_matrix_ref = torch.matmul(uncompressed, rhs_matrix)
        output_matrix = spmmv2_bf16(nonzeros, rhs_matrix, metadata)
            
        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, rtol=1e-9))
    
    def test(self):
        dense_matrix = torch.randn(size=(batch_size, feat_in), dtype=dtype, device="cuda")
        nonzeros_ref, _, metadata_ref = bdense2sparse_gold(dense_matrix)
        nonzeros, metadata = bdense2sparse(dense_matrix)

        self.assertTrue(torch.allclose(nonzeros_ref, nonzeros, rtol=1e-9))
        self.assertTrue(torch.allclose(metadata_ref, metadata, rtol=1e-9))



class TestSpMM(unittest.TestCase):
    def test(self):
        dense_matrix = torch.randn(size=(batch_size, feat_in), dtype=dtype, device="cuda")
        rhs_matrix = torch.randn(size=(feat_in, feat_out), dtype=dtype, device="cuda")
        nonzeros, uncompressed, metadata = bdense2sparse_gold(dense_matrix)
        output_matrix_ref = torch.matmul(uncompressed, rhs_matrix)
        output_matrix = spmmv2_bf16(nonzeros, rhs_matrix, metadata)
            
        self.assertTrue(torch.allclose(output_matrix, output_matrix_ref, rtol=1e-1))

if __name__ == '__main__':
    unittest.main()