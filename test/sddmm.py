import torch
import unittest
from sptrain.meta import bdense2sparse_gold
from sptrain.sddmm import sddmm_bf16_ntn, sddmm_f16_ntn
import torch.nn.functional as F
import math

batch_size = 4
head = 2
bs = int(batch_size / head)
sequence_length = 4096
embedding = 64


bf16 = torch.bfloat16
half = torch.float16


class TestSDDMM(unittest.TestCase):

    def test_sddmm_bf16(self):
        mask = None
        query = torch.randn(size=(sequence_length, embedding), dtype=bf16, device="cuda")
        key = torch.randn(size=(sequence_length, embedding), dtype=bf16, device="cuda")
        dense_matrix_ref = torch.matmul(query, key.t())

        nonzeros_ref, uncompressed, metadata_ref = bdense2sparse_gold(dense_matrix_ref, False)
        nonzeros, metadata = sddmm_bf16_ntn(query, key, mask, 1.)

        self.assertTrue(torch.ne(metadata, metadata_ref).sum() / metadata.numel() < 5e-3)
        self.assertTrue(torch.ne(nonzeros, nonzeros_ref).sum() / nonzeros.numel() < 1e-3)

    
    def test_sddmm_f16(self):
        mask = None
        query = torch.randn(size=(sequence_length, embedding), dtype=half, device="cuda")
        key = torch.randn(size=(sequence_length, embedding), dtype=half, device="cuda")
        dense_matrix_ref = torch.matmul(query, key.t())

        nonzeros_ref, uncompressed, metadata_ref = bdense2sparse_gold(dense_matrix_ref, False)
        nonzeros, metadata = sddmm_f16_ntn(query, key, mask, 1.)

        self.assertTrue(torch.ne(metadata, metadata_ref).sum() / metadata.numel() < 5e-3)
        self.assertTrue(torch.ne(nonzeros, nonzeros_ref).sum() / nonzeros.numel() < 1e-3)
    
    def test_batched_sddmm_bf16(self):
        mask = None
        query = torch.randn(size=(batch_size, sequence_length, embedding), dtype=bf16, device="cuda")
        key = torch.randn(size=(batch_size, sequence_length, embedding), dtype=bf16, device="cuda")
        dense_matrix_ref = torch.bmm(query, key.transpose(1, 2))

        nonzeros_ref, uncompressed, metadata_ref = bdense2sparse_gold(dense_matrix_ref, False)
        nonzeros, metadata = sddmm_bf16_ntn(query, key, mask, 1.)

        self.assertTrue(torch.ne(metadata, metadata_ref).sum() / metadata.numel() < 5e-3)
        self.assertTrue(torch.ne(nonzeros, nonzeros_ref).sum() / nonzeros.numel() < 1e-3)

    
    def test_batched_sddmm_f16(self):
        mask = None
        query = torch.randn(size=(batch_size, sequence_length, embedding), dtype=half, device="cuda")
        key = torch.randn(size=(batch_size, sequence_length, embedding), dtype=half, device="cuda")
        dense_matrix_ref = torch.bmm(query, key.transpose(1, 2))

        nonzeros_ref, uncompressed, metadata_ref = bdense2sparse_gold(dense_matrix_ref, False)
        nonzeros, metadata = sddmm_f16_ntn(query, key, mask, 1.)

        self.assertTrue(torch.ne(metadata, metadata_ref).sum() / metadata.numel() < 5e-3)
        self.assertTrue(torch.ne(nonzeros, nonzeros_ref).sum() / nonzeros.numel() < 1e-3)
    
    ## Test masked cases
    def test_sddmm_bf16_mask(self):
        prob = torch.ones(size=(1, 1, 1, sequence_length), dtype=bf16, device='cuda') * 0.2
        mask = torch.bernoulli(prob) * -1e16
        query = torch.randn(size=(sequence_length, embedding), dtype=bf16, device="cuda")
        key = torch.randn(size=(sequence_length, embedding), dtype=bf16, device="cuda")
        dense_matrix_ref = (torch.matmul(query, key.t()).view(1, -1, sequence_length, sequence_length) + mask).view(1, sequence_length, sequence_length)

        alpha = 1./ math.sqrt(embedding)
        nonzeros_ref, uncompressed, metadata_ref = bdense2sparse_gold(dense_matrix_ref * alpha, False)
        nonzeros, metadata = sddmm_bf16_ntn(query, key, mask, alpha)

        self.assertTrue(torch.ne(metadata, metadata_ref).sum() / metadata.numel() < 5e-3)
        self.assertTrue(torch.ne(nonzeros, nonzeros_ref).sum() / nonzeros.numel() < 1e-3)

    
    def test_sddmm_f16_mask(self):
        prob = torch.ones(size=(1, 1, 1, sequence_length), dtype=half, device='cuda') * 0.2
        mask = torch.bernoulli(prob) * -1e16
        query = torch.randn(size=(sequence_length, embedding), dtype=half, device="cuda")
        key = torch.randn(size=(sequence_length, embedding), dtype=half, device="cuda")
        dense_matrix_ref = (torch.matmul(query, key.t()).view(1, -1, sequence_length, sequence_length) + mask).view(1, sequence_length, sequence_length)

        alpha = 1./ math.sqrt(embedding)
        nonzeros_ref, uncompressed, metadata_ref = bdense2sparse_gold(dense_matrix_ref * alpha, False)
        nonzeros, metadata = sddmm_f16_ntn(query, key, mask, alpha)

        self.assertTrue(torch.ne(metadata, metadata_ref).sum() / metadata.numel() < 5e-3)
        self.assertTrue(torch.ne(nonzeros, nonzeros_ref).sum() / nonzeros.numel() < 1e-3)
    
    def test_batched_sddmm_bf16_mask(self):
        prob = torch.ones(size=(bs, 1, 1, sequence_length), dtype=bf16, device='cuda') * 0.2
        mask = torch.bernoulli(prob) * -1e16
        query = torch.randn(size=(batch_size, sequence_length, embedding), dtype=bf16, device="cuda")
        key = torch.randn(size=(batch_size, sequence_length, embedding), dtype=bf16, device="cuda")
        dense_matrix_ref = (torch.bmm(query, key.transpose(1, 2)).view(bs, -1, sequence_length, sequence_length) + mask).view(batch_size, sequence_length, sequence_length)

        alpha = 1./ math.sqrt(embedding)
        nonzeros_ref, uncompressed, metadata_ref = bdense2sparse_gold(dense_matrix_ref * alpha, False)
        nonzeros, metadata = sddmm_bf16_ntn(query, key, mask, alpha)

        self.assertTrue(torch.ne(metadata, metadata_ref).sum() / metadata.numel() < 5e-3)
        self.assertTrue(torch.ne(nonzeros, nonzeros_ref).sum() / nonzeros.numel() < 1e-3)

    
    def test_batched_sddmm_f16_mask(self):
        prob = torch.ones(size=(bs, 1, 1, sequence_length), dtype=half, device='cuda') * 0.15
        mask = torch.bernoulli(prob) * -1e16

        query = torch.randn(size=(batch_size, sequence_length, embedding), dtype=half, device="cuda")
        key = torch.randn(size=(batch_size, sequence_length, embedding), dtype=half, device="cuda")
        ddd = torch.bmm(query, key.transpose(1, 2))
        dense_matrix_ref = (torch.bmm(query, key.transpose(1, 2)).view(bs, -1, sequence_length, sequence_length) + mask).view(batch_size, sequence_length, sequence_length)

        alpha = 1./ math.sqrt(embedding)
        nonzeros_ref, uncompressed, metadata_ref = bdense2sparse_gold(dense_matrix_ref * alpha, False)
        nonzeros, metadata = sddmm_f16_ntn(query, key, mask, alpha)

        self.assertTrue(torch.ne(metadata, metadata_ref).sum() / metadata.numel() < 5e-3)
        self.assertTrue(torch.ne(nonzeros, nonzeros_ref).sum() / nonzeros.numel() < 1e-3)

if __name__ == '__main__':
    unittest.main()