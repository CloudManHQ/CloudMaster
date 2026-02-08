# 监督学习 (Supervised Learning)

监督学习是利用已知标签的训练数据来学习输入到输出的映射函数。

## 1. 核心模型与原理 (Core Models & Principles)

### 线性与逻辑回归 (Linear & Logistic Regression)
- **线性回归 (Linear Regression)**: 用于预测连续值。
    - **损失函数**: 均方误差 (Mean Squared Error, MSE)。
    - **优化**: 闭式解 (Normal Equation) 或 随机梯度下降 (SGD)。
- **逻辑回归 (Logistic Regression)**: 用于二分类问题。
    - **核心**: Sigmoid 函数将输出映射到 (0, 1)。
    - **损失函数**: 交叉熵损失 (Cross-Entropy Loss)。

### 集成学习 (Ensemble Learning)
- **Bagging (自助聚合法)**:
    - **随机森林 (Random Forest)**: 通过构建多个决策树并取平均/投票来降低方差。
- **Boosting (提升法)**:
    - **XGBoost**: 高效实现的梯度提升决策树 (GBRT)，引入 L1/L2 正则化。
    - **LightGBM**: 基于直方图的决策树算法，支持并行训练。
    - **CatBoost**: 专门优化类别特征处理。
- **来源**: [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)

## 2. 评估与泛化 (Evaluation & Generalization)
- **偏差-方差权衡 (Bias-Variance Tradeoff)**: 解决欠拟合与过拟合。
- **验证集与交叉验证 (K-Fold Cross-Validation)**。

## 3. 来源参考
- [Introduction to Statistical Learning (ISL)](https://www.statlearning.com/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/supervised_learning.html)
