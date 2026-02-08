# 强化学习基础 (RL Foundations)

强化学习通过试错 (Trial and Error) 学习最优行为策略。

## 1. 核心数学模型 (Mathematical Model)

### 马尔可夫决策过程 (Markov Decision Process, MDP)
- **五个要素**: 状态集合 $S$、动作集合 $A$、转移概率 $P$、奖励函数 $R$、折扣因子 $\gamma$。
- **贝尔曼方程 (Bellman Equation)**: 定义了值函数的递归关系。

## 2. 核心算法分类
- **基于价值 (Value-based)**: Q-Learning, SARSA。
- **基于策略 (Policy-based)**: REINFORCE (策略梯度基础)。
- **演员-评论家 (Actor-Critic)**: 结合价值函数与策略参数化。

## 3. 来源参考
- [Reinforcement Learning: An Introduction - Sutton & Barto](http://incompleteideas.net/book/the-book-2nd.html)
