from cgi import print_arguments
import torch
import unittest
from sptrain.meta import bdense2sparse_gold
from sptrain.sddmm_target import sddmm_target_bf16_ntn, sddmm_target_f16_ntn
import torch.nn.functional as F
import math

batch_size = 4
head = 2
bs = int(batch_size / head)
sequence_length = 4096
embedding = 64


bf16 = torch.bfloat16
half = torch.float16


class TestSDDMMTarget(unittest.TestCase):
    
    def test_sddmm_f16_output_predicated(self):
        mask = None
        input = torch.randn(size=(28*128, 1024), dtype=half, device="cuda")
        weight = torch.randn(size=(32320, 1024), dtype=half, device="cuda")
        bias = torch.randn(size=(32320,), dtype=half, device="cuda")
        dense_matrix_ref = F.linear(input, weight, bias)  #torch.matmul(input, weight.t()) 

        target = torch.randint(low=0, high=32320, size=(28 * 128,1), dtype=torch.int64, device="cuda")

        src = torch.ones_like(target).to(half) * 2345
        
        dense_matrix_ref.scatter_add_(dim=1, index=target, src=src)

        nonzeros_ref, uncompressed, metadata_ref = bdense2sparse_gold(dense_matrix_ref, False)

        nonzeros, metadata, target_sp = sddmm_target_f16_ntn(input, weight, target, bias, 1.)

        nonzeros = nonzeros.squeeze().scatter_add_(dim=1, index=target_sp, src=src)

        # TODO: inner index to the rows are not correct

        self.assertTrue(torch.ne(metadata, metadata_ref).sum() / metadata.numel() < 5e-3)

        self.assertTrue(torch.ge(nonzeros - nonzeros_ref,1000).sum() / target.numel() < 1e-3)
        self.assertTrue(torch.ne(nonzeros, nonzeros_ref).sum() / nonzeros.numel() < 2e-3)
    

if __name__ == '__main__':
    unittest.main()