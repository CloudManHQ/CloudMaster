# 数据结构与算法 (Data Structures & Algorithms)

高效的算法是实现大规模 AI 模型训练与推理的基础。

## 1. AI 相关的核心算法 (Core Algorithms for AI)

### 计算图与反向传播 (Computational Graphs & Backpropagation)
- **图论基础**: 神经网络被表示为有向无环图 (Directed Acyclic Graph, DAG)。
- **拓扑排序 (Topological Sort)**: 确定前向传播顺序。
- **自动微分 (Automatic Differentiation)**: 通过计算图上的链式法则实现梯度传递。
- **来源**: [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)

### 搜索与优化算法 (Search & Optimization)
- **贪心搜索 (Greedy Search)** 与 **束搜索 (Beam Search)**: 大模型推理时的解码策略。
- **动态规划 (Dynamic Programming)**: 用于强化学习中的贝尔曼方程求解。

## 2. 数据结构在模型中的应用 (Data Structures in Models)

### 向量与矩阵存储
- **稠密张量 (Dense Tensors)**: 标准多维数组。
- **稀疏张量 (Sparse Tensors)**: 处理包含大量零元素的权重矩阵，优化内存和计算。

### 空间索引 (Spatial Indexing)
- **KD-Tree** 与 **球树 (Ball Tree)**: 用于高效的近邻搜索 (K-NN)。
- **HNSW (Hierarchical Navigable Small World)**: 现代向量数据库中检索增强生成 (RAG) 的核心算法。
- **来源**: [Efficient and Robust Approximate Nearest Neighbor Search using Hierarchical Navigable Small World Graphs](https://arxiv.org/abs/1603.09320)

## 3. 推荐资源 (Recommended Resources)
- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/)
- [CS 61B: Data Structures (UC Berkeley)](https://sp21.datastructur.es/)
