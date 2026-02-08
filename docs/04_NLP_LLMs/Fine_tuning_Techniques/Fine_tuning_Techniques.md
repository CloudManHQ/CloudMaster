# 微调技术 (Fine-tuning Techniques)

针对特定下游任务优化预训练模型的关键技术。

## 1. 参数高效微调 (PEFT)

### LoRA (Low-Rank Adaptation)
- **原理**: 冻结预训练权重，通过低秩矩阵分解学习权重的增量更新。
- **公式**: $\Delta W = A \times B$。
- **来源**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### QLoRA
- **原理**: 结合 4-bit 量化与 LoRA，大幅降低显存占用。
- **来源**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

## 2. 软提示词微调 (Soft Prompting)
- **P-Tuning / Prefix Tuning**: 在输入序列前加入可学习的 Embedding。

## 3. 来源参考
- [Hugging Face PEFT Library Documentation](https://huggingface.co/docs/peft/index)
