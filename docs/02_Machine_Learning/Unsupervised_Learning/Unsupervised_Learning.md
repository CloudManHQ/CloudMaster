# 无监督学习 (Unsupervised Learning)

无监督学习旨在发现未标记数据中的潜在结构和模式。

## 1. 聚类分析 (Clustering Analysis)

### K-Means 聚类
- **原理**: 通过迭代寻找 $K$ 个簇中心，最小化簇内误差平方和 (WCSS)。
- **术语**: 质心 (Centroid)、惯性 (Inertia)、肘部法则 (Elbow Method)。
- **来源**: [Scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html#k-means)

### 层次聚类与 DBSCAN
- **DBSCAN**: 基于密度的聚类算法，能够发现任意形状的簇并识别噪声点。
- **术语**: 核心点 (Core Points)、边界点 (Border Points)、噪声 (Noise)。

## 2. 降维技术 (Dimensionality Reduction)

### 主成分分析 (Principal Component Analysis, PCA)
- **原理**: 寻找数据方差最大的正交轴，通过特征分解实现。
- **来源**: [A Tutorial on Principal Component Analysis - Jonathon Shlens](https://arxiv.org/abs/1404.1100)

### t-SNE 与 UMAP
- **t-SNE**: 适用于高维数据可视化的非线性降维技术。
- **UMAP**: 基于黎曼几何的流形学习算法，兼顾局部与全局结构。

## 3. 推荐资源
- [Elements of Statistical Learning - Chapter 14](https://web.stanford.edu/~hastie/ElemStatLearn/)
