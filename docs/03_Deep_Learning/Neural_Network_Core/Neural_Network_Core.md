# 神经网络核心 (Neural Network Core)

神经网络是深度学习的基础架构，通过层级化特征提取实现复杂模式识别。

## 1. 架构组件 (Architectural Components)

### 神经元与层 (Neurons & Layers)
- **全连接层 (Fully Connected/Linear Layer)**: 线性变换 $y = Wx + b$。
- **激活函数 (Activation Functions)**: 
    - **ReLU (Rectified Linear Unit)**: 解决梯度消失问题。
    - **GELU (Gaussian Error Linear Unit)**: Transformer 中的主流选择。
- **归一化 (Normalization)**:
    - **Batch Normalization**: 加速深层网络训练。
    - **Layer Normalization**: NLP 任务中的核心组件。

## 2. 训练算法 (Training Algorithms)

### 反向传播 (Backpropagation)
- **原理**: 基于链式法则 (Chain Rule) 的梯度计算。
- **梯度消失与爆炸 (Vanishing & Exploding Gradients)**: 深层网络的挑战。
- **来源**: [Deep Learning - Chapter 6: Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html)

## 3. 推荐资源
- [Deep Learning Specialization - Andrew Ng](https://www.deeplearning.ai/courses/deep-learning-specialization/)
- [PyTorch Neural Network Tutorial](https://pytorch.org/tutorials/beginner/nn_tutorial.html)
