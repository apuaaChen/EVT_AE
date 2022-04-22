import torch
import unittest
from sptrain.meta import bdense2sparse_gold
from sptrain.sddmm import sddmm_bf16_ntn, sddmm_f16_ntn
import torch.nn.functional as F

sequence_length = 4096
embedding = 64


bf16 = torch.bfloat16
half = torch.float16


class TestSDDMM(unittest.TestCase):

    def test_sddmm_bf16(self):
        query = torch.randn(size=(sequence_length, embedding), dtype=bf16, device="cuda")
        key = torch.randn(size=(sequence_length, embedding), dtype=bf16, device="cuda")
        dense_matrix_ref = torch.matmul(query, key.t())
        dense_matrix = sddmm_bf16_ntn(query, key)

        # print(dense_matrix_ref[0][0:32] - dense_matrix[0][0:32])

        self.assertTrue(torch.allclose(dense_matrix, dense_matrix_ref, rtol=1e-9))
    
    def test_sddmm_f16(self):
        query = torch.randn(size=(sequence_length, embedding), dtype=half, device="cuda")
        key = torch.randn(size=(sequence_length, embedding), dtype=half, device="cuda")
        dense_matrix_ref = torch.matmul(query, key.t())
        dense_matrix = sddmm_f16_ntn(query, key)

        self.assertTrue(torch.allclose(dense_matrix, dense_matrix_ref, rtol=1e-9))

if __name__ == '__main__':
    unittest.main()