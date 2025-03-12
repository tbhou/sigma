# Σ-GenAI for Open Research

This repo collects some latest research work of Generative AI. It provides simple implementations to understand the ideas and some follow-up discussions to inspire future work.

## Attention

### Native Sparse Attention

[Native Sparse Attention](https://arxiv.org/abs/2502.11089) (NSA) is proposed by DeepSeek for efficient language models. The techniques can also be applied to vision generation. Since video generation is often trained by diffusion or flow matching with full attention, the sliding attention in NSA can be skipped. Token compression and selection can be done in 1D or 3D, depending on the latent space.

**Follow-up**

Token compression provides an overview of the entire sequence. Uncompressed token blocks are then selected based on their attentions scores. A recent work [DART](https://www.arxiv.org/abs/2502.11494) proposes to consider duplication in token selection. It first selects a fewer pivotal tokens, and then similar token to the pivotal ones are ignored in the token selection. This allows to select more diverse tokens. In another work [DivPrune](https://arxiv.org/abs/2503.02175), the problem is formulated a Max–Min Diversity Problem (MMDP). It'll be interesting to see how these techniques work for generation tasks.

## MoE

### DeepSeekMoE

[DeepSeekMoE](https://arxiv.org/pdf/2401.06066) uses fine-grained experts and shared experts on top of standard MoE. An MLP layer typically uses a large intermediate dimension, e.g. 6 times of input dimension. DeepSeekMoE uses a smaller intermediate dimension to get more experts without increasing computation. Using fine-grained experts enhances the combinatorial flexibility of activated experts.

**Follow-up**

MoE is efficient at inference when batch size is 1. If batch size is greater than 1, more experts will be activated, and it can take more memories than a dense model. To address this problem, ByteDance releases [Comet](https://arxiv.org/pdf/2502.19811), an efficient infrastructure for MoE.