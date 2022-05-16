import torch
import torch.nn.functional as F
import unittest
from sptrain.softmax import softmax


class TestSoftmax(unittest.TestCase):

    def test_softmax(self):

        dense_matrix = torch.randn(size=(28*128, 32320), dtype=torch.float16, device="cuda")
        target = torch.randint(low=0, high=32320, size=(28*128, 1), dtype=torch.int64, device="cuda")
        confidence=0.9
        bias = 0.1

        src = torch.ones_like(target).to(torch.float16) * confidence

        out_ref = F.softmax(dense_matrix, dim=-1) + bias
        out_ref.scatter_add_(dim=1, index=target, src=src)
        out = softmax(dense_matrix, -1, bias, target, confidence)

        self.assertTrue(torch.allclose(out, out_ref, rtol=5e-2))


if __name__ == '__main__':
    unittest.main()