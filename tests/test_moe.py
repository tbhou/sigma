import torch
import unittest

from model.moe import SigmaGate, SigmaMoE

# python3 -m unittest tests/test_moe.py


class TestMoE(unittest.TestCase):

    def test_gate(self):
        B, N, K, dim = 2, 32, 2, 8
        gate = SigmaGate(dim, 8, K)
        x = torch.randn(B, N, dim)
        idx, weight = gate(x)
        self.assertEqual(idx.shape, torch.Size([B * N, K]))
        self.assertEqual(weight.shape, torch.Size([B * N, K]))

    def test_moe(self):
        B, N, K, dim = 1, 4, 2, 8
        moe = SigmaMoE(dim, 2, 4, K)
        x = torch.randn(B, N, dim)
        # training
        moe.training = True
        y = moe(x)
        self.assertEqual(y.shape, torch.Size([B, N, dim]))
        # inference
        moe.training = False
        y = moe(x)
        self.assertEqual(y.shape, torch.Size([B, N, dim]))
