# 线性代数 (Linear Algebra)

线性代数是深度学习的语言，从数据表示到变换，几乎所有的机器学习操作都可以归结为矩阵运算。

## 1. 核心概念 (Core Concepts)

### 张量 (Tensors)
- **标量 (Scalar)**: 0阶张量。
- **向量 (Vector)**: 1阶张量，数据的有序集合。
- **矩阵 (Matrix)**: 2阶张量，用于线性变换。
- **多维张量 (N-D Tensor)**: 深度学习中的基础数据结构，如彩色图像表示为 3阶张量 $(H, W, C)$。
- **来源**: [PyTorch Tensors Documentation](https://pytorch.org/docs/stable/tensors.html)

### 矩阵运算与线性变换 (Matrix Operations & Linear Transformations)
- **矩阵乘法 (Matrix Multiplication)**: 复合线性变换的基础。
- **逆矩阵 (Inverse Matrix)** 与 **行列式 (Determinant)**。
- **正交矩阵 (Orthogonal Matrix)**: 保持向量长度不变，在模型权重初始化中至关重要。

### 特征值分解 (Eigenvalue Decomposition, EVD)
- **定义**: $A\mathbf{v} = \lambda\mathbf{v}$。
- **物理意义**: 特征向量表示变换的主轴，特征值表示在该轴上的缩放程度。

### 奇异值分解 (Singular Value Decomposition, SVD)
- **公式**: $A = U\Sigma V^T$。
- **应用**:
    - **主成分分析 (Principal Component Analysis, PCA)**: 用于降维。
    - **低秩近似 (Low-Rank Approximation)**: 模型压缩和矩阵填充。
- **来源**: [The Fundamental Theorem of Linear Algebra - Gilbert Strang](https://web.mit.edu/18.06/www/essays/newpaper_7.pdf)

## 2. 在 AI 中的应用 (Applications in AI)

- **权重表示**: 神经网络的每一层本质上都是权重矩阵与输入向量的乘法加偏移。
- **注意力机制**: Transformer 中的 $Q, K, V$ 矩阵计算。
- **优化算法**: 牛顿法中的黑塞矩阵 (Hessian Matrix)。

## 3. 推荐资源 (Recommended Resources)
- [Linear Algebra Done Right - Sheldon Axler](https://linear.axler.net/)
- [3Blue1Brown - Essence of Linear Algebra (Visual Learning)](https://www.3blue1brown.com/topics/linear-algebra)
