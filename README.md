# Code Snippets for Video Generation

Thie repo collects some code snippets for video generation. It is a playground to follow recent research in the field by impelementating the core ideas.

## Native Sparse Attention

[Native Sparse Attention](https://arxiv.org/abs/2502.11089) (NSA) is proposed by DeepSeek for efficient langugae models. The techniques can also be applied to vision generation. Since video generation is often trained by diffusion or flow matching with full attention, the sliding attention in NSA can be skipped. The ideas of token compression and selection have been adopted in vision domain. I implmented both 1D and 3D versions for block splitting. 