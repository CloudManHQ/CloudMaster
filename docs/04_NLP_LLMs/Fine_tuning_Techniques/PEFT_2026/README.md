# PEFT 2026 (参数高效微调)

## 文档导航

| 文档 | 内容 | 适用读者 |
|------|------|----------|
| [PEFT_2026.md](./PEFT_2026.md) | LoRA/QLoRA/DoRA 2026最佳实践 | 实战学习 |

## 核心方法

| 方法 | 显存需求 | 质量 | 特点 |
|------|---------|------|------|
| LoRA | 2x A100 | 95% | 标准方法 |
| QLoRA | 48GB | 93% | 消费级GPU可训70B |
| DoRA | 2x A100 | 98% | 质量最优 |
| rsLoRA | 2x A100 | 96% | 支持高rank |

## 超参数速查

```python
config = {
    "r": 16,          # rank: 8-64
    "alpha": 32,      # 2*r
    "dropout": 0.05,  # 0.05-0.1
    "lr": 2e-4,       # 1e-4 ~ 2e-4
    "epochs": 1-3,    # 避免过拟合
}
```

## 一句话总结

> **PEFT让大模型微调不再"烧钱"** — 2026年的QLoRA可以在单张消费级显卡上微调70B参数模型。

---

## 参考

- [HuggingFace PEFT](https://github.com/huggingface/peft)
- [Unsloth](https://github.com/unslothai/unsloth) - 加速微调
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
