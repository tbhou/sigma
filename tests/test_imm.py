import torch
from torch import nn
import unittest

from pipeline.imm import IMM, ConsineScheduler



class DummyModel(nn.Module):

    def forward(self, x, s, t):
        del s, t
        return torch.rand(*x.shape)
    

class TestIMM(unittest.TestCase):

    def setUp(self):
        self.imm = IMM(DummyModel(), 4, ConsineScheduler())

    def test_train_step(self):
        x = torch.rand(8, 16, 8)
        loss = self.imm.train_step(x)
        print(loss)

    def test_inference_step(self):
        x_shape = torch.Size((1, 16, 8))
        x = self.imm.inference_step(4, x_shape)
        self.assertEqual(x.shape, x_shape)
