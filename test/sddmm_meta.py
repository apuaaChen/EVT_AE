import torch
import unittest
from sptrain.meta import bdense2sparse_gold
from sptrain.sddmm_meta import sddmm_meta_bf16_ntn, sddmm_meta_f16_ntn

batch_size = 4
head = 2
bs = int(batch_size / head)
sequence_length = 4096
embedding = 64


bf16 = torch.bfloat16
half = torch.float16


class TestSDDMMMeta(unittest.TestCase):

    def test_batched_sddmm_meta_f16(self):
        query = torch.randn(size=(batch_size, sequence_length, embedding), dtype=half, device="cuda")
        key = torch.randn(size=(batch_size, sequence_length, embedding), dtype=half, device="cuda")
        dense_matrix_ref = torch.bmm(query, key.transpose(1, 2))
        nonzeros_ref, uncompressed, metadata = bdense2sparse_gold(dense_matrix_ref, False)

        nonzeros = sddmm_meta_f16_ntn(query, key, metadata, 1.)

        self.assertTrue(torch.ne(nonzeros, nonzeros_ref).sum() / nonzeros.numel() < 1e-3)
    
    def test_batched_sddmm_meta_bf16(self):
        query = torch.randn(size=(batch_size, sequence_length, embedding), dtype=bf16, device="cuda")
        key = torch.randn(size=(batch_size, sequence_length, embedding), dtype=bf16, device="cuda")
        dense_matrix_ref = torch.bmm(query, key.transpose(1, 2))
        nonzeros_ref, uncompressed, metadata = bdense2sparse_gold(dense_matrix_ref, False)

        nonzeros = sddmm_meta_bf16_ntn(query, key, metadata, 1.)

        self.assertTrue(torch.ne(nonzeros, nonzeros_ref).sum() / nonzeros.numel() < 1e-3)

if __name__ == '__main__':
    unittest.main()