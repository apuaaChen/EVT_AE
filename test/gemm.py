import torch
import unittest
from sptrain.gemm import gemm_bf16_nnn


batch_size = 4096
feat_in = 1024
feat_out = 2048


dtype = torch.bfloat16

class TestGemm(unittest.TestCase):

    def test_gemm(self):
        lhs_matrix = torch.randn(size=(batch_size, feat_in), dtype=dtype, device="cuda")
        rhs_matrix = torch.randn(size=(feat_in, feat_out), dtype=dtype, device="cuda")

        output_ref = torch.matmul(lhs_matrix, rhs_matrix)
        output = gemm_bf16_nnn(lhs_matrix, rhs_matrix)

        self.assertTrue(torch.allclose(output_ref, output, atol=0.5))


if __name__ == '__main__':
    unittest.main()