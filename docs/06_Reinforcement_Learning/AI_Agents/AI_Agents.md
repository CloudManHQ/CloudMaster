# AI 智能体 (AI Agents)

智能体是能够自主规划、记忆并使用工具完成复杂任务的系统。

## 1. 智能体架构 (Agent Architecture)

### 核心组件
- **大脑 (Brain)**: 大语言模型 (LLM) 作为核心推理机。
- **规划 (Planning)**: 任务分解 (Chain-of-Thought)、自我反思 (Self-Reflection)。
- **记忆 (Memory)**: 短期记忆（Context）、长期记忆（向量数据库）。
- **工具使用 (Tool Use)**: 调用外部 API、代码解释器。

## 2. 知名项目与框架
- **AutoGPT / BabyAGI**: 早期的自主智能体尝试。
- **Generative Agents**: 模拟人类社会行为。
- **框架**: LangGraph, CrewAI, Microsoft AutoGen。

## 3. 来源参考
- [LLM Powered Autonomous Agents - Lilian Weng](https://lilianweng.github.io/posts/2023-06-23-agent/)
