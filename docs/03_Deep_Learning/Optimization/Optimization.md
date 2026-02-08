# 训练优化 (Optimization)

高效的优化算法是深度学习模型收敛的关键。

## 1. 优化器 (Optimizers)

### 随机梯度下降家族 (SGD Family)
- **SGD with Momentum**: 引入动量项加速收敛并减少震荡。
- **Nesterov Accelerated Gradient (NAG)**: 预测未来梯度的改进动量法。

### 自适应学习率算法 (Adaptive Learning Rates)
- **Adam (Adaptive Moment Estimation)**: 结合了一阶矩（动量）和二阶矩（方差）估计。
- **AdamW**: 修正了 Adam 在权重衰减 (Weight Decay) 实现上的错误。
- **来源**: [Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101)

## 2. 正则化技术 (Regularization)
- **Dropout**: 随机丢弃神经元以防止过拟合。
- **权重衰减 (Weight Decay)**: L2 正则化的工程实现。
- **早停法 (Early Stopping)**: 基于验证集性能停止训练。

## 3. 来源参考
- [An overview of gradient descent optimization algorithms - Sebastian Ruder](https://arxiv.org/abs/1609.04747)
