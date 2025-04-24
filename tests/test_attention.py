import torch

import unittest

from model.attention import Attention, SparseAttention, SparseAttention3D, MultiTokenAttention


class TestAttention(unittest.TestCase):

    def test_attention(self):
        dim, head = 16, 2
        attn = Attention(dim, head)
        x = torch.randn(1, 32, dim)
        y = attn(x)
        self.assertEqual(x.shape, y.shape)

        kv = torch.randn(1, 4, dim)
        y = attn(x, kv)
        self.assertEqual(x.shape, y.shape)

        freqs_cis = torch.randn(32, dim // (head * 2))
        y = attn(x, freqs_cis=freqs_cis)
        self.assertEqual(x.shape, y.shape)

    def test_sparse_attention(self):
        dim, head = 16, 4
        attn = SparseAttention(dim, head, 4, 3)
        x = torch.randn(1, 32, dim)
        freqs_cis = torch.randn(32, dim // (head * 2))
        y = attn(x, freqs_cis=freqs_cis)
        self.assertEqual(x.shape, y.shape)

    def test_sparse_attention_3d(self):
        dim, head = 16, 4
        attn = SparseAttention3D(dim, head, [2, 2, 2], 2)
        x = torch.randn(1, 64, dim)
        freqs_cis = torch.randn(64, dim // (head * 2))
        y = attn(x, freqs_cis=freqs_cis, latent_shape=[4, 4, 4])
        self.assertEqual(x.shape, y.shape)

    def test_mta_attention(self):
        dim, heads = 16, 4
        kernel_size = (3, 3, 2)  # q, k, head
        attn = MultiTokenAttention(dim, heads, kernel_size=kernel_size)
        x = torch.randn(1, 32, dim)
        y = attn(x)
        self.assertEqual(x.shape, y.shape)


if __name__ == '__main__':
    unittest.main()