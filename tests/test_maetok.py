import torch
import unittest

from model.maetok import SigmaEncoder, SigmaDecoder


class TestMAEToK(unittest.TestCase):

    def test_encoder(self):
        dim, num_latent_token = 16, 4
        enc = SigmaEncoder(dim, 2, 3, 8, num_latent_token, 0.5)
        x = torch.randn(1, 3, 4, 8)  # BCHW
        tokens = enc(x)
        self.assertEqual(tokens.shape, torch.Size([1, num_latent_token, dim]))

    def test_decoder(self):
        H, W, num_image_token, num_latent_token, dim = 4, 8, 8, 4, 16
        dec = SigmaDecoder(dim, 2, 3, num_image_token, num_latent_token)
        z = torch.randn(1, num_latent_token, dim)  # BLD
        image = dec(z, [H, W])
        self.assertEqual(image.shape, torch.Size([1, 3, H, W]))
