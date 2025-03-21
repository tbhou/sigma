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


## Auto-Encoder

Modern image and video generation models use CNN based auto-encoders (AEs), with the following limitations
- Uniform compression. AEs compress an image or a video uniformly in spatial and temporal domain.
- Not aligned with representations for understanding.
- Efficiency. Interestingly, CNN-AEs require large dimensions at high resolutions, which are not efficient.

Some recent work starts to use ViTs for auto-encoders, including [TiTok](https://arxiv.org/abs/2406.07550), [ViTok](https://arxiv.org/pdf/2501.09755), and [MAEToK](https://arxiv.org/pdf/2502.03444). If we can encode an image or a video to a 1D sequence of tokens, in a coarse-to-fine manner, auto-regressive methods may fit better, and the latent space can be aligned with text for better multi-modality understanding. The work ([Semanticist](https://arxiv.org/pdf/2503.08685)) moves one step ahead towards this direction.

### MAEToK

MAEToK uses [MAE](https://arxiv.org/pdf/2111.06377) to learn latent tokens for generation. In its implementation, the token masking is different from the original MAE. Specifically, MAE discards masked tokens, and only sends remaining tokens for transformers. MAEToK sends all pixel tokens to transformers with masked tokens replaced by a learnable token. I followed the original MAE masking in my implementation. Another thing to be noted is that MAEToK builds AE and VAE on top of VQ. 


## Diffusion Sampling

### Inductive Moment Matching

[Inductive Moment Matching (IMM)](https://arxiv.org/abs/2503.07565) is a new class of generative models. The training is based on mathematical induction. For s < r < t, it matches two distributions from r, t to s by minimizing their divergence. To make the training stable, it uses stable sample-based divergence estimators, i.e. moment matching. With a single-stage training, it can be sampled efficiently with one or a few steps. The official implementation doesn't provide training code, and there are some unspecified parameters in the paper. My implementation tries to follow the algorithms provided in the paper, and aims to help people to understand the work.