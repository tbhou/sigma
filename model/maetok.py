import torch

from torch import nn

from einops import rearrange


class VectorQuantizer(nn.Module):
    """Dummy vector quantizer."""

    def forward(self, x):
        return x


class SigmaEncoder(nn.Module):
    """Masked ViT encoder.
    
    It follows the original MAE, such that masked tokens are discarded.
    It outputs latent tokens. Image tokens are discarded.
    """

    def __init__(self, dim, patch_size, in_channels, num_image_token, num_latent_token, mask_ratio):
        super().__init__()
        self.dim = dim
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_image_token, dim), requires_grad=False)
        self.num_image_token = num_image_token
        self.num_latent_token = num_latent_token
        self.mask_ratio = mask_ratio
        self.latent_token = nn.Parameter(torch.zeros(1, num_latent_token, dim))
        self.latent_pos_embed = nn.Parameter(torch.zeros(1, num_latent_token, dim), requires_grad=False)
        self.init_weights()

    def init_weights(self):
        # TODO
        pass
    
    def random_masking(self, x):
        B, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked

    def forward(self, x):
        x = self.patch_embed(x)  
        x = x.flatten(2).transpose(1, 2)  # BDHW -> BDL -> BLD
        x = x + self.pos_embed  # no class token
        
        # mask tokens
        if self.training:
            x = self.random_masking(x)

        # latent token
        z = self.latent_token.expand(x.shape[0], -1, -1)
        z = z + self.latent_pos_embed
        x = torch.cat([x, z], dim=1)
        
        # TODO: apply transformer layers
        
        return x[:, -self.num_latent_token:]  # latent tokens


class SigmaDecoder(nn.Module):
    """ViT Decoder."""

    def __init__(self, dim, patch_size, in_channels, num_image_token, num_latent_token):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.num_image_token = num_image_token
        self.to_pixel = nn.Linear(dim, in_channels * patch_size * patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_image_token, dim), requires_grad=False)
        self.latent_pos_embed = nn.Parameter(torch.zeros(1, num_latent_token, dim), requires_grad=False)

    def unpatchify(self, x, orig_shape):
        x = self.to_pixel(x)
        H, W = orig_shape
        P, C = self.patch_size, self.in_channels
        LH, LW = H // P, W // P
        x = rearrange(x, 'b (h w) (p q c) -> b c (h p) (w q)', h=LH, w=LW, p=P, q=P, c=C)
        return x

    def forward(self, z, orig_shape):
        x = self.mask_token.expand(z.shape[0], self.num_image_token, -1)
        x = x + self.pos_embed
        z = z + self.latent_pos_embed
        x = torch.cat([x, z], dim=1)
        # TODO: apply transformer layers
        return self.unpatchify(x[:, :self.num_image_token], orig_shape)
        

class SigmaTok(nn.Module):
    """VQ auto-encodeder."""

    def __init__(self, dim, patch_size, in_channels, num_image_token, num_latent_token, mask_ratio, codebook_dim):
        super().__init__()
        self.encoder = SigmaEncoder(dim, patch_size, in_channels, num_image_token, num_latent_token, mask_ratio)
        self.decoder = SigmaDecoder(dim, patch_size, in_channels, num_image_token, num_latent_token)
        self.quant_proj = nn.Linear(dim, codebook_dim)
        self.quantizer = VectorQuantizer()
        self.dequant_proj = nn.Linear(codebook_dim, dim)

    def encode(self, x):
        x = self.encoder(x)
        h = self.quant_proj(x)
        h = self.quantizer(h)
        return h
    
    def decode(self, z, orig_shape):
        z = self.dequant_proj(z)
        x = self.decoder(z, orig_shape)
        return x