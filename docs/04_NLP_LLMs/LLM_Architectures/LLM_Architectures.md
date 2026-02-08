# 大语言模型架构 (LLM Architectures)

解析现代 LLM 的主流设计模式与架构演进。

## 1. 预训练范式 (Pre-training Paradigms)

### 仅解码器架构 (Decoder-only)
- **代表**: GPT-3, GPT-4, LLaMA, PaLM。
- **特点**: 因果语言建模 (Causal Language Modeling)，适合文本生成。
- **来源**: [Language Models are Few-Shot Learners (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)

### 仅编码器架构 (Encoder-only)
- **代表**: BERT, RoBERTa。
- **特点**: 掩码语言建模 (Masked Language Modeling)，适合理解任务。

### 编码器-解码器架构 (Encoder-Decoder)
- **代表**: T5, BART。
- **特点**: 适合翻译、摘要等序列到序列 (Seq2Seq) 任务。

## 2. 混合专家模型 (Mixture of Experts, MoE)
- **原理**: 仅激活模型中一小部分参数（专家）进行计算，极大提升规模。
- **代表**: Mixtral 8x7B, GPT-4 (据推测)。
- **来源**: [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

## 3. 推荐资源
- [LLM Survey - A Survey of Large Language Models](https://arxiv.org/abs/2303.18223)
