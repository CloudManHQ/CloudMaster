# 08 AI 伦理、安全与对齐 (Ethics, Safety & Alignment)

本章探讨 AI 系统的可信度与责任性，涵盖价值对齐技术（RLHF/DPO）、AI 安全与红队测试（对抗攻击/提示词注入）。随着 AI 能力增强，确保系统安全、公平、可控变得至关重要。

## 学习路径 (Learning Path)

```
    ┌──────────────────────┐
    │  价值对齐             │
    │  Value Alignment     │
    │  (RLHF/DPO)          │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  AI 安全与红队        │
    │  AI Safety &         │
    │  Red Teaming         │
    │  (对抗/防御)          │
    └──────────────────────┘
```

## 内容索引 (Content Index)

| 主题 | 难度 | 描述 | 文档链接 |
|------|------|------|---------|
| 价值对齐 (Value Alignment) | 进阶 | RLHF、DPO、奖励建模，让 AI 输出符合人类偏好 | [Value_Alignment.md](./Value_Alignment/Value_Alignment.md) |
| AI 安全与红队 (AI Safety & Red Teaming) | 实战 | 对抗样本、提示词注入、越狱攻击、安全护栏，防御恶意使用 | [AI_Safety_RedTeaming.md](./AI_Safety_RedTeaming/AI_Safety_RedTeaming.md) |

## 前置知识 (Prerequisites)

- **必修**: [大语言模型架构](../04_NLP_LLMs/LLM_Architectures/LLM_Architectures.md)（理解 LLM 行为）
- **必修**: [微调技术](../04_NLP_LLMs/Fine_tuning_Techniques/Fine_tuning_Techniques.md)（RLHF 是微调的一种）
- **推荐**: [深度强化学习](../06_Reinforcement_Learning/Deep_RL/Deep_RL.md)（RLHF 中的 PPO）
- **推荐**: [提示词工程](../04_NLP_LLMs/Prompt_Engineering/)（理解越狱攻击）

## 关键术语速查 (Key Terms)

- **RLHF (Reinforcement Learning from Human Feedback)**: 基于人类反馈的强化学习，训练奖励模型后用 PPO 优化策略
- **DPO (Direct Preference Optimization)**: 直接偏好优化，绕过奖励模型的对齐方法
- **奖励模型 (Reward Model)**: 学习人类偏好的评分模型，用于 RLHF
- **对齐 (Alignment)**: 确保 AI 行为符合人类价值观和意图
- **红队测试 (Red Teaming)**: 模拟攻击者测试 AI 系统的安全性和鲁棒性
- **对抗样本 (Adversarial Examples)**: 精心构造的输入，欺骗模型产生错误输出
- **提示词注入 (Prompt Injection)**: 通过特殊提示词绕过 AI 安全限制
- **越狱 (Jailbreaking)**: 诱导模型输出违反安全政策的内容
- **安全护栏 (Safety Guardrails)**: 检测和阻止有害输入/输出的机制
- **公平性 (Fairness)**: 确保 AI 系统不歧视特定群体，输出无偏见

---
*Last updated: 2026-02-10*
