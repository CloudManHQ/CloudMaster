# Transformer 革命 (Transformer Revolution)

Transformer 架构彻底改变了 NLP 领域，成为了现代大语言模型 (LLM) 的基石。

## 1. 核心机制 (Core Mechanisms)

### 注意力机制 (Attention Mechanism)
- **自注意力 (Self-Attention)**: 捕捉序列内部的远距离依赖。
- **多头注意力 (Multi-Head Attention)**: 允许模型在不同子空间学习信息。
- **缩放点积注意力 (Scaled Dot-Product Attention)**: $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$。

### 架构细节
- **位置编码 (Positional Encoding)**: 为模型提供序列顺序信息（如 RoPE 旋转位置编码）。
- **残差连接 (Residual Connections)** 与 **层归一化 (LayerNorm)**。

## 2. 来源与影响
- **论文**: [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- **影响**: 开启了从 BERT 到 GPT 的预训练大模型时代。

## 3. 推荐学习资源
- [The Illustrated Transformer - Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Course on Transformers](https://huggingface.co/learn/nlp-course/chapter1/1)
