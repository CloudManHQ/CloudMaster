# 概率论与数理统计 (Probability & Statistics)

概率论提供了处理 AI 系统中不确定性 (Uncertainty) 的数学框架。

## 1. 核心概念 (Core Concepts)

### 贝叶斯理论 (Bayesian Theory)
- **先验概率 (Prior Probability)**: $P(A)$。
- **似然度 (Likelihood)**: $P(B|A)$。
- **后验概率 (Posterior Probability)**: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$。
- **应用**: 朴素贝叶斯分类器、贝叶斯神经网络。

### 常见概率分布 (Common Distributions)
- **高斯分布 (Gaussian/Normal Distribution)**: 深度学习中权重初始化的常见假设，中心极限定理的基础。
- **伯努利分布 (Bernoulli Distribution)**: 二分类问题的理论基础。
- **多项式分布 (Multinomial Distribution)**: 多分类问题与语言模型中 Token 采样。

### 估计理论 (Estimation Theory)
- **最大似然估计 (Maximum Likelihood Estimation, MLE)**: 通过最大化观测数据出现的概率来估计参数。
- **最大后验概率估计 (Maximum A Posteriori, MAP)**: 引入先验分布，相当于在损失函数中增加正则化项 (如 L2 对应高斯先验)。
- **来源**: [Deep Learning Book - Chapter 3: Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html)

## 2. 统计推断与模型 (Statistical Inference & Models)

### 熵与信息论 (Entropy & Information Theory)
- **香农熵 (Shannon Entropy)**: 度量不确定性。
- **交叉熵 (Cross-Entropy)**: 深度学习中最常用的损失函数，衡量两个分布的差异。
- **KL 散度 (Kullback-Leibler Divergence)**: 用于衡量模型预测分布与真实分布的偏差，常见于 VAE。

## 3. 推荐资源 (Recommended Resources)
- [Probability Theory: The Logic of Science - E.T. Jaynes](https://www.cambridge.org/core/books/probability-theory/973F8D76F2912DCC228B12270922900B)
- [Harvard Stat 110: Introduction to Probability](https://statistics.fas.harvard.edu/people/joseph-k-blitzstein)
