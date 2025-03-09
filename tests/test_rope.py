import torch

import unittest

from model.rope import apply_rope, get_1d_rope, get_nd_rope, get_meshgrid_nd


class TestRope(unittest.TestCase):

    def test_get_1d_rope(self):
        l, dim = 16, 4
        pos = torch.arange(16).float()
        emb = get_1d_rope(dim, pos)
        self.assertEqual(emb.shape, torch.Size([l, dim // 2]))

    def test_get_nd_rope(self):
        dim_list = [2, 2, 2]
        dim = sum(dim_list)
        T, H, W = 8, 16, 16
        sizes = [T, H, W]
        emb = get_nd_rope(dim_list, sizes)
        self.assertEqual(emb.shape, torch.Size([T * H * W, dim // 2]))

    def test_apply_rope(self):
        B, S, H, D = 1, 16, 2, 4
        q, k = torch.randn(B, S, H, D), torch.randn(B, S, H, D)
        freqs_cis = torch.randn(S, D // 2)
        xq, xk = apply_rope(q, k, freqs_cis)
        self.assertEqual(xq.shape, q.shape)
        self.assertEqual(xk.shape, k.shape)


if __name__ == '__main__':
    unittest.main()