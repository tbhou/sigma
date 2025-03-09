# Open Research of Generative AI

This repo collects some latest research work of Generative AI. It provides simple implementations to understand the ideas and some follow-up discussions to inspire future work.

## Efficient Modeling

### Native Sparse Attention

[Native Sparse Attention](https://arxiv.org/abs/2502.11089) (NSA) is proposed by DeepSeek for efficient langugae models. The techniques can also be applied to vision generation. Since video generation is often trained by diffusion or flow matching with full attention, the sliding attention in NSA can be skipped. Token compression and selection can be done in 1D or 3D, depending on the latent space.

**Follow-ups**

Token compression provides an overview of the entire sequence. Uncompressed token blocks are then selected based on their attentions scores. A recent work [DART](https://www.arxiv.org/abs/2502.11494) proposes to condier duplication in token selection. It first selects a fewer pivotal tokens, and then similar token to the pivotal ones are ignored in the token selection. This allows to select more diversed tokens. In another work [DivPrune](https://arxiv.org/abs/2503.02175), the problem is formulated a Max–Min Diversity Problem (MMDP). It'll be interesting to see how these techques work for generation tasks.

## TBD