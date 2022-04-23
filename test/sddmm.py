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

        nonzeros_ref, uncompressed, metadata_ref = bdense2sparse_gold(dense_matrix_ref, False)
        nonzeros, metadata = sddmm_bf16_ntn(query, key)

        self.assertTrue(torch.ne(metadata, metadata_ref).sum() / metadata.numel() < 5e-3)
        self.assertTrue(torch.ne(nonzeros, nonzeros_ref).sum() / nonzeros.numel() < 1e-3)

    
    def test_sddmm_f16(self):
        query = torch.randn(size=(sequence_length, embedding), dtype=half, device="cuda")
        key = torch.randn(size=(sequence_length, embedding), dtype=half, device="cuda")
        dense_matrix_ref = torch.matmul(query, key.t())

        nonzeros_ref, uncompressed, metadata_ref = bdense2sparse_gold(dense_matrix_ref, False)
        nonzeros, metadata = sddmm_f16_ntn(query, key)

        self.assertTrue(torch.ne(metadata, metadata_ref).sum() / metadata.numel() < 5e-3)
        self.assertTrue(torch.ne(nonzeros, nonzeros_ref).sum() / nonzeros.numel() < 1e-3)

if __name__ == '__main__':
    unittest.main()