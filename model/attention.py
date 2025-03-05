from copy import deepcopy
from typing import Tuple

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from model.rope import apply_rope


class Attention(nn.Module):
    """Basic multi-head attention."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout_rate: float = 0.0
    ):
        super().__init__()

        self.heads = heads
        self.dim_head = dim // heads
        self.dropout_rate = dropout_rate

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)

        self.split_heads = Rearrange('b n (h d) -> b n h d', d = self.dim_head)
        self.merge_heads = Rearrange('b n h d -> b n (h d)')

        self.out_proj = nn.Linear(dim, dim, bias = False)

    def attend(self, q, k, v):
        scale = q.shape[-1] ** -0.5
        # for selected attention, kv are different for each q token
        same_kv = len(q.shape) == len(k.shape)
        qk_pattern = 'bqhd,bkhd->bhqk' if same_kv else 'bqhd,bqkhd->bhqk'
        score = torch.einsum(qk_pattern, q, k) * scale
        score = torch.softmax(score, dim=-1)
        score = torch.dropout(score, p=self.dropout_rate, train=self.training)
        sv_pattern = 'bhqk,bkhd->bqhd' if same_kv else 'bhqk,bqkhd->bqhd'
        y = torch.einsum(sv_pattern, score, v)
        return y, score

    def forward(self, x, kv=None, freqs_cis=None):
        # x: [b, n, d]
        q = self.to_q(x)

        if kv is None:
            kv = x
        k = self.to_k(kv)
        v = self.to_v(kv)

        # q: [b, n, h, d]
        q, k, v = map(self.split_heads, (q, k, v))

        # positional embedding
        if freqs_cis is not None:
            q, k = apply_rope(q, k, freqs_cis)

        y = self.attend(q, k, v)[0]
        y = self.merge_heads(y)

        return self.out_proj(y)


class SparseAttention(Attention):
    """Sparse multi-head attention.
    
    An implementation of "Native Sparse Attention" for video generation. It
    computes full attention without masking, and skips sliding attention. It
    assumes the latent is a 1D sequence of tokens that can be split to blocks
    in 1D.
    """
    def __init__(
        self,
        dim: int,
        heads: int,
        block_size: int,
        num_selected_blocks: int,
        mlp_expansion: int = 1,
    ):
        super().__init__(dim, heads)

        self.block_size = block_size
        self.intrablock_pe = nn.Parameter(
            torch.zeros(self.block_size, self.heads, self.dim_head)
        )

        compress_dim = self.block_size * self.dim_head
        compress_dim_hidden = int(mlp_expansion * compress_dim)

        compress_mlp = nn.Sequential(
            Rearrange('b w n h d -> b w h (n d)'),
            nn.Linear(compress_dim, compress_dim_hidden),
            nn.ReLU(),
            nn.Linear(compress_dim_hidden, self.dim_head),
        )  # [b, w, h, d]

        self.k_compress = deepcopy(compress_mlp)
        self.v_compress = deepcopy(compress_mlp)

        assert num_selected_blocks > 0
        self.num_selected_blocks = num_selected_blocks

        # combine strategy: b n d -> b n sh -> b n s
        self.to_combine = nn.Sequential(
            nn.Linear(dim, num_selected_blocks * heads),
            nn.Sigmoid(),
            Rearrange('b n (s h) -> b n s h', h = heads)
        )

    def split_blocks(self, x, latent_shpae=None):
        return rearrange(x, 'b (w n) h d -> b w n h d', n = self.block_size)

    def add_block_pe(self, x):
        num_blocks = x.shape[1]
        pe = repeat(self.intrablock_pe, 'n h d -> w n h d', w = num_blocks)
        return x + pe

    def forward(self, x, freqs_cis=None, latent_shape=None):
        # x: [b, n, d]
        num_blocks = x.shape[1] // self.block_size

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(self.split_heads, (q, k, v))

        # split to blocks and compress
        k_blocks = self.split_blocks(k, latent_shape)  # (b, w, n, h, d)
        v_blocks = self.split_blocks(v, latent_shape)

        k_blocks = self.add_block_pe(k_blocks)
        v_blocks = self.add_block_pe(v_blocks)

        ck = self.k_compress(k_blocks)  # (b, w, h, d)
        cv = self.v_compress(v_blocks)

        # compressed attention
        compressed_attn, score = self.attend(q, ck, cv)

        # selected attention
        if freqs_cis is not None:
            sq, sk = apply_rope(q, k, freqs_cis)

        num_selected = min(self.num_selected_blocks, num_blocks)
        selected_indices = score.topk(num_selected, dim = -1)[1]  # (b, h, q, s)

        sk = self.split_blocks(sk, latent_shape)  # (b, w, n, h, d)
        sv = self.split_blocks(v, latent_shape)  # (b, w, n, h, d)
        sk = repeat(sk, 'b w n h d -> b q w n h d', q = selected_indices.shape[2])
        sv = repeat(sv, 'b w n h d -> b q w n h d', q = selected_indices.shape[2])

        selected_indices = repeat(
            selected_indices,
            'b h q s -> b q s n h d',
            n = sk.shape[-3],
            d = sk.shape[-1],
        )
        sk = sk.gather(2, selected_indices)  # b q s n h d
        sv = sv.gather(2, selected_indices)

        sk = rearrange(sk, 'b q s n h d -> b q (s n) h d')
        sv = rearrange(sv, 'b q s n h d -> b q (s n) h d')
        selected_attn = self.attend(sq, sk, sv)[0]  # [b q h d]

        # combine strategies
        combined_x = self.to_combine(x)
        combined_attn = torch.stack([compressed_attn, selected_attn])
        y = torch.einsum('bnsh,sbnhd->bnhd', combined_x, combined_attn)

        y = self.merge_heads(y)  # (b, n, d)
        return self.out_proj(y)
    

class SparseAttention3D(SparseAttention):
    """3D sparse multi-head attention.
    
    It assumes the latent is a 1D sequence of tokens flattened from 3D, and
    it splits blocks in 3D along three axes.
    """
    def __init__(
        self,
        dim: int,
        heads: int,
        block_shape: Tuple[int, int, int],
        num_selected_blocks: int,
        mlp_expansion: int = 1,
    ):
        bf, bh, bw = block_shape
        super().__init__(
            dim, 
            heads, 
            bf * bh * bw,
            num_selected_blocks,
            mlp_expansion,
        )
        self.block_shape = block_shape
        self.intrablock_pe = nn.Parameter(
            torch.zeros(bf, bh, bw, self.heads, self.dim_head)
        )

    def split_blocks(self, x, latent_shape):
        F, H, W = latent_shape
        b, n, h, d = x.shape
        assert n == F * H * W
        x = x.view(b, F, H, W, h, d)
        bf, bh, bw = self.block_shape
        x = rearrange(
            x,
            'b (nf bf) (nh bh) (nw bw) h d -> b (nf nh nw) (bf bh bw) h d',
            bf=bf,
            bh=bh,
            bw=bw,
        )
        return x
    
    def add_block_pe(self, x):
        num_blocks = x.shape[1]        
        pe = repeat(self.intrablock_pe, 'bf bh bw h d -> w (bf bh bw) h d', w = num_blocks)
        return x + pe
