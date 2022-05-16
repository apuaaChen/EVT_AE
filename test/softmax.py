import torch
import torch.nn.functional as F
import unittest
from sptrain.softmax import softmax


class TestSoftmax(unittest.TestCase):

    def test_softmax(self):

        dense_matrix = torch.randn(size=(28*128, 32320), dtype=torch.float16, device="cuda")
        bias = 0.1

        out_ref = F.softmax(dense_matrix, dim=-1) + bias
        out = softmax(dense_matrix, -1, bias)

        self.assertTrue(torch.allclose(out, out_ref, rtol=5e-2))


if __name__ == '__main__':
    unittest.main()