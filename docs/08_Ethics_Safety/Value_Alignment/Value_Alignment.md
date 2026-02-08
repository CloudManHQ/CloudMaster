# 价值对齐 (Value Alignment)

价值对齐旨在确保 AI 系统的行为与人类的目标、偏好和伦理原则一致。

## 1. 核心技术 (Core Techniques)

### RLHF (基于人类反馈的强化学习)
- **阶段 1: SFT (监督微调)**: 学习模仿人类回答。
- **阶段 2: 奖励模型 (Reward Model)**: 训练一个评估回复质量的模型。
- **阶段 3: PPO 优化**: 使用强化学习最大化奖励。

### 替代方案
- **DPO (Direct Preference Optimization)**: 无需显式奖励模型的轻量化对齐方案。
- **KTO (Kahneman-Tversky Optimization)**: 基于前景理论的对齐方法。

## 2. 对齐指标 (Alignment Dimensions)
- **有用性 (Helpfulness)**。
- **诚实性 (Honesty)**。
- **无害性 (Harmlessness)**。

## 3. 来源参考
- [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
