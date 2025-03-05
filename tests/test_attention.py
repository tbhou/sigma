import torch

import unittest

from model.attention import Attention, SparseAttention, SparseAttention3D


class TestAttention(unittest.TestCase):

    def test_attention(self):
        dim = 16
        attn = Attention(dim, 2)
        x = torch.randn(1, 32, dim)
        y = attn(x)
        self.assertEqual(x.shape, y.shape)

        kv = torch.randn(1, 4, dim)
        y = attn(x, kv)
        self.assertEqual(x.shape, y.shape)

        freqs_cis = torch.randn(32, dim // 2)
        y = attn(x, freqs_cis=freqs_cis)
        self.assertEqual(x.shape, y.shape)

    def test_sparse_attention(self):
        dim = 16
        attn = SparseAttention(dim, 4, 4, 2)
        x = torch.randn(1, 32, dim)
        freqs_cis = torch.randn(32, dim // 2)
        y = attn(x, freqs_cis=freqs_cis)
        self.assertEqual(x.shape, y.shape)

    def test_sparse_attention_3d(self):
        dim = 16
        attn = SparseAttention3D(dim, 4, [2, 2, 2], 2)
        x = torch.randn(1, 64, dim)
        freqs_cis = torch.randn(64, dim // 2)
        y = attn(x, freqs_cis=freqs_cis, latent_shape=[4, 4, 4])
        self.assertEqual(x.shape, y.shape)


if __name__ == '__main__':
    unittest.main()