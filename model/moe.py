# Code modified from 
# https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/modeling_deepseek.py
#

import math
import torch
import torch.nn.functional as F

from torch import nn


class SigmaMLP(nn.Module):
    """Gated MLP layer.
    
    Args:
        dim: Dimension of input tokens.
        intermediate: Dimension of intermediate layers. It's typically greater than dim. But for MoE,
            this can be smaller.

    Returns:
        y: MLP result.
    """

    def __init__(self, dim, intermediate):
        super().__init__()
        self.dim = dim
        self.intermediate = intermediate
        self.up_proj = nn.Linear(dim, self.intermediate, bias=False)
        self.gate_proj = nn.Linear(dim, self.intermediate, bias=False)
        self.down_proj = nn.Linear(self.intermediate, dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.up_proj(x)
        gate = self.act(self.gate_proj(x))
        y = y * gate
        y = self.down_proj(y)
        return y
    

class SigmaGate(nn.Module):
    """Gate dispatcher of experts.
    
    Args:
        dim: Dimension of input tokens.
        n_experts: The total number of experts.
        k_experts: The number of experts per token.

    Returns:
        topk_idx: Indices of topk experts.
        topk_weight: Normalized weight (scores) for topk experts.
    """

    def __init__(self, dim, n_experts, k_experts):
        super().__init__()
        self.n_experts = n_experts
        self.k_experts = k_experts
        self.weight = nn.Parameter(torch.empty((n_experts, dim)))
        # initialize weight
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        x = x.view(-1, x.shape[-1])  # [bn, d]
        logits = F.linear(x, self.weight, bias=None)  # [bn, e]
        scores = logits.softmax(dim=-1)
        # [bn, k]
        topk_weight, topk_idx = torch.topk(scores, k=self.k_experts, dim=-1, sorted=False)
        # normalize
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight  / denominator
        return topk_idx, topk_weight


class SigmaMoE(nn.Module):
    """DeepSeek MoE with fine-grained experts and shared experts."""
    
    def __init__(self, dim, intermediate, n_experts, k_experts, s_experts: int = 1):
        super().__init__()
        self.n_experts = n_experts  # total number of experts
        self.experts = nn.ModuleList([SigmaMLP(dim, intermediate) for _ in range(n_experts)])
        self.k_experts = k_experts  # topk experts
        self.gate = SigmaGate(dim, n_experts, k_experts)
        self.shared_experts = SigmaMLP(dim, intermediate * s_experts)

    def forward(self, x):
        identity = x
        orig_shape = x.shape  # [b, n, d]
        topk_idx, topk_weight = self.gate(x)  # [bn, k]
        x = x.view(-1, x.shape[-1])  # [bn, d]
        flat_topk_idx = topk_idx.view(-1)  # [bnk]
        if self.training:
            x = x.repeat_interleave(self.k_experts, dim=0)  # [bnk, d]
            y = torch.empty_like(x)  # [bnk, d]
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i])
            y = y.view(*topk_weight.shape, -1)  # [bn, k, d]
            y = (y * topk_weight.unsqueeze(-1)).sum(dim=1)  # [bn, d]
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1))
        y = y.view(*orig_shape)  # [b, n, d]
        # shared experts
        y = y + self.shared_experts(identity)
        return y

    def moe_infer(self, x, flat_idx, flat_weights):
        # x: [bn, d]
        expert_cache = torch.zeros_like(x)
        idx = flat_idx.argsort()  # [bnk], max=bnk
        tokens_per_expert = flat_idx.bincount().cpu().numpy().cumsum(0)  # [e]
        token_idx = idx // self.k_experts  # max=bn
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idx[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]  # [t, d], t: # of tokens for this expert
            expert_out = expert(expert_tokens)
            expert_out = expert_out * flat_weights[idx[start_idx:end_idx]]
            exp_token_idx = exp_token_idx.view(-1, 1).repeat(1, x.shape[-1])
            expert_cache.scatter_reduce_(0, exp_token_idx, expert_out, reduce='sum')
        return expert_cache
