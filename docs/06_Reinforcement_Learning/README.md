# 06 强化学习与智能体 (Reinforcement Learning & Agents)

本章涵盖强化学习的完整技术栈，从数学基础（MDP/贝尔曼方程）到深度强化学习算法（DQN/PPO），再到自主智能体架构（推理规划/工具使用）。这是构建自主决策系统的核心技术。

## 学习路径 (Learning Path)

```
    ┌──────────────────────┐
    │  强化学习基础         │
    │  RL Foundations      │
    │  (MDP/Bellman)       │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  深度强化学习         │
    │  Deep RL             │
    │  (DQN/PPO/SAC)       │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  AI Agents           │
    │  智能体架构           │
    │  (推理/记忆/工具)     │
    └──────────────────────┘
```

## 内容索引 (Content Index)

| 主题 | 难度 | 描述 | 文档链接 |
|------|------|------|---------|
| 强化学习基础 (RL Foundations) | 入门 | MDP、贝尔曼方程、策略梯度、价值函数，RL 数学框架 | [RL_Foundations.md](./RL_Foundations/RL_Foundations.md) |
| 深度强化学习 (Deep RL) | 进阶 | DQN、PPO、SAC、离线 RL，结合神经网络的 RL 算法 | [Deep_RL.md](./Deep_RL/Deep_RL.md) |
| AI Agents (智能体) | 实战 | ReAct、长期记忆、工具使用、多智能体系统，自主决策架构 | [AI_Agents.md](./AI_Agents/AI_Agents.md) |

## 前置知识 (Prerequisites)

- **必修**: [概率统计](../01_Fundamentals/Probability_Statistics/Probability_Statistics.md)（理解 MDP 和期望计算）
- **必修**: [神经网络核心](../03_Deep_Learning/Neural_Network_Core/Neural_Network_Core.md)（深度 RL 中的函数逼近）
- **推荐**: [优化与正则化](../03_Deep_Learning/Optimization/Optimization.md)（稳定 RL 训练）
- **可选**: [大语言模型架构](../04_NLP_LLMs/LLM_Architectures/LLM_Architectures.md)（理解 LLM 驱动的智能体）

## 关键术语速查 (Key Terms)

- **MDP (Markov Decision Process)**: 马尔可夫决策过程，RL 的数学建模框架
- **贝尔曼方程 (Bellman Equation)**: 描述价值函数递归关系，RL 理论核心
- **策略 (Policy)**: 从状态到动作的映射，π(a|s)
- **价值函数 (Value Function)**: 评估状态或动作的长期回报 V(s) 或 Q(s,a)
- **Q-Learning**: 无模型价值学习算法，学习最优动作价值函数
- **DQN (Deep Q-Network)**: 结合深度学习的 Q-Learning，使用经验回放和目标网络
- **PPO (Proximal Policy Optimization)**: 策略梯度算法，通过裁剪更新稳定训练
- **Actor-Critic**: 结合策略和价值函数的 RL 架构
- **ReAct (Reasoning + Acting)**: 推理与执行交织的智能体范式
- **Multi-Agent RL**: 多智能体强化学习，处理协作与竞争场景

---
*Last updated: 2026-02-10*
