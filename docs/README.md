# AI Guru 知识库 (AI Guru Knowledge Base)

欢迎来到 **AI Guru** 知识库。这是一份系统化的 AI 知识体系指南，旨在为开发者、研究人员和 AI 爱好者提供从底层原理到前沿应用的深度解析。每个主题都包含直觉解释、数学原理、代码实战和面试高频问题，兼顾初学者理解和专家参考。

## 如何使用本知识库 (How to Use)

### 按角色选择学习路径

**零基础入门者** — 从头开始建立 AI 知识体系：
```
01 基础理论 → 02 机器学习 → 03 深度学习 → 04 NLP/LLM (按兴趣选) 或 05 CV
```

**转行工程师** — 有编程基础，快速切入 AI 工程：
```
03 深度学习 → 04 NLP/LLM → 07 AI 工程化 → 06 AI Agents
```

**研究者/面试准备** — 深入理论，拓宽前沿视野：
```
01 基础理论(数学) → 03 深度学习 → 04 NLP/LLM → 06 强化学习 → 08 伦理安全
```

### 知识依赖关系图

```
01 基础理论 ──→ 02 机器学习 ──→ 03 深度学习 ──┬──→ 04 NLP & LLMs
 (数学/CS)       (经典算法)      (神经网络)    │      (Transformer/LLM)
                                              │
                                              ├──→ 05 计算机视觉
                                              │      (CNN/分割/生成)
                                              │
                                              └──→ 06 强化学习 & 智能体
                                                     (RL/Agent)
                                                        │
                            07 AI 工程化 ←──────────────┘
                             (部署/RAG/MLOps)
                                  │
                            08 伦理与安全
                             (对齐/红队)
```

---

## 知识体系大纲 (Knowledge System Taxonomy)

### [01 基础理论 (Fundamentals)](./01_Fundamentals/README.md)

AI 的数学基石与计算机科学基础。

| 主题 | 难度 | 核心内容 |
|------|------|---------|
| [线性代数](./01_Fundamentals/Linear_Algebra/Linear_Algebra.md) | 入门 | 张量、矩阵运算、EVD/SVD、LoRA 的数学基础 |
| [概率论与数理统计](./01_Fundamentals/Probability_Statistics/Probability_Statistics.md) | 入门 | 贝叶斯定理、MLE/MAP、信息论（熵/交叉熵/KL散度） |
| [数据结构与算法](./01_Fundamentals/Data_Structures_Algorithms/Data_Structures_Algorithms.md) | 进阶 | 计算图、自动微分、Beam Search、HNSW 向量检索 |
| [分布式系统](./01_Fundamentals/Distributed_Systems/Distributed_Systems.md) | 进阶 | All-Reduce、数据/模型/流水线并行、ZeRO 优化 |

**参考来源**: [Deep Learning Book](https://www.deeplearningbook.org/) | [Mathematics for ML](https://mml-book.github.io/)

---

### [02 经典机器学习 (Classical Machine Learning)](./02_Machine_Learning/README.md)

传统机器学习算法与数据处理技术。

| 主题 | 难度 | 核心内容 |
|------|------|---------|
| [监督学习](./02_Machine_Learning/Supervised_Learning/Supervised_Learning.md) | 入门 | 线性回归、SVM、决策树、XGBoost/LightGBM |
| [无监督学习](./02_Machine_Learning/Unsupervised_Learning/Unsupervised_Learning.md) | 入门 | K-Means、DBSCAN、PCA、t-SNE/UMAP |
| [特征工程](./02_Machine_Learning/Feature_Engineering/Feature_Engineering.md) | 实战 | 标准化/编码、特征交叉/选择、时间序列特征、Feature Store |

**参考来源**: [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) | [PRML - Bishop](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)

---

### [03 深度学习基础 (Deep Learning Foundations)](./03_Deep_Learning/README.md)

神经网络核心原理与训练优化技术。

| 主题 | 难度 | 核心内容 |
|------|------|---------|
| [神经网络核心](./03_Deep_Learning/Neural_Network_Core/Neural_Network_Core.md) | 入门 | MLP、反向传播推导、激活函数、权重初始化 |
| [训练优化](./03_Deep_Learning/Optimization/Optimization.md) | 进阶 | SGD/Adam/AdamW、学习率调度、混合精度训练(AMP) |

**参考来源**: [PyTorch Tutorials](https://pytorch.org/tutorials/) | [CS231n (Stanford)](http://cs231n.stanford.edu/)

---

### [04 自然语言处理与大模型 (NLP & LLMs)](./04_NLP_LLMs/README.md)

从序列模型到大语言模型的完整技术栈。

| 主题 | 难度 | 核心内容 |
|------|------|---------|
| [序列模型](./04_NLP_LLMs/Sequence_Models/Sequence_Models.md) | 入门 | RNN、LSTM 门控机制、GRU、Seq2Seq、Attention 起源 |
| [Transformer 革命](./04_NLP_LLMs/Transformer_Revolution/Transformer_Revolution.md) | 进阶 | Self-Attention 推导、位置编码、KV Cache、Flash Attention |
| [大语言模型架构](./04_NLP_LLMs/LLM_Architectures/LLM_Architectures.md) | 进阶 | GPT/LLaMA/Gemini 对比、MoE、Scaling Laws |
| [微调技术](./04_NLP_LLMs/Fine_tuning_Techniques/Fine_tuning_Techniques.md) | 实战 | LoRA/QLoRA、RLHF/DPO/ORPO 对齐技术 |
| [提示词工程](./04_NLP_LLMs/Prompt_Engineering/Prompt_Engineering.md) | 实战 | Zero/Few-shot、CoT/ToT、结构化输出、安全设计 |

**参考来源**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | [Hugging Face Course](https://huggingface.co/learn/nlp-course/)

---

### [05 计算机视觉 (Computer Vision)](./05_Computer_Vision/README.md)

图像理解、分割、生成与多模态融合。

| 主题 | 难度 | 核心内容 |
|------|------|---------|
| [图像分类与检测](./05_Computer_Vision/Image_Classification_Detection/Image_Classification_Detection.md) | 入门 | CNN 演进(ResNet/EfficientNet)、ViT、YOLO 系列 |
| [图像分割](./05_Computer_Vision/Segmentation/Segmentation.md) | 进阶 | U-Net、DeepLab、Mask R-CNN、SAM |
| [多模态视觉](./05_Computer_Vision/Multimodal_Vision/Multimodal_Vision.md) | 进阶 | CLIP、LLaVA、BLIP-2、GPT-4V |
| [生成模型](./05_Computer_Vision/Generative_Models/Generative_Models.md) | 进阶 | GAN、Diffusion Models、Stable Diffusion、ControlNet |

**参考来源**: [Deep Residual Learning](https://arxiv.org/abs/1512.03385) | [Segment Anything](https://arxiv.org/abs/2304.02643)

---

### [06 强化学习与智能体 (RL & Agents)](./06_Reinforcement_Learning/README.md)

从经典强化学习到现代 AI Agent 系统。

| 主题 | 难度 | 核心内容 |
|------|------|---------|
| [强化学习基础](./06_Reinforcement_Learning/RL_Foundations/RL_Foundations.md) | 入门 | MDP、贝尔曼方程、Q-Learning、探索-利用权衡 |
| [深度强化学习](./06_Reinforcement_Learning/Deep_RL/Deep_RL.md) | 进阶 | DQN、PPO、SAC、TD3、Model-based RL |
| [AI 智能体](./06_Reinforcement_Learning/AI_Agents/AI_Agents.md) | 实战 | Agent 架构、ReAct/Reflexion、Tool Calling、多智能体 |

**参考来源**: [Sutton & Barto RL Book](http://incompleteideas.net/book/the-book-2nd.html) | [OpenAI Spinning Up](https://spinningup.openai.com/)

---

### [07 AI 工程化与 MLOps (AI Engineering & MLOps)](./07_AI_Engineering/README.md)

将 AI 模型转化为生产力的工程实践。

| 主题 | 难度 | 核心内容 |
|------|------|---------|
| [模型部署与推理](./07_AI_Engineering/Deployment_Inference/Deployment_Inference.md) | 实战 | vLLM/TensorRT-LLM、量化(AWQ/GPTQ)、模型蒸馏 |
| [RAG 系统](./07_AI_Engineering/RAG_Systems/RAG_Systems.md) | 实战 | 文档分块、向量检索、混合搜索、GraphRAG |
| [MLOps 流水线](./07_AI_Engineering/MLOps_Pipeline/MLOps_Pipeline.md) | 实战 | 实验跟踪(MLflow)、数据版本(DVC)、CI/CD for ML |
| [模型评估](./07_AI_Engineering/Model_Evaluation/Model_Evaluation.md) | 入门 | 分类/回归/生成指标、LLM 评估基准、统计检验 |

**参考来源**: [vLLM](https://github.com/vllm-project/vllm) | [Pinecone Learning Center](https://www.pinecone.io/learn/)

---

### [08 AI 伦理、安全与对齐 (Ethics, Safety & Alignment)](./08_Ethics_Safety/README.md)

确保 AI 系统安全、公平、可控。

| 主题 | 难度 | 核心内容 |
|------|------|---------|
| [价值对齐](./08_Ethics_Safety/Value_Alignment/Value_Alignment.md) | 进阶 | RLHF/DPO、数据偏见检测、公平性指标、Constitutional AI |
| [AI 安全与红队](./08_Ethics_Safety/AI_Safety_RedTeaming/AI_Safety_RedTeaming.md) | 实战 | 提示注入防御、Guardrails 系统、红队测试方法论、法规速查 |

**参考来源**: [Anthropic: Core Views on AI Safety](https://www.anthropic.com/news/core-views-on-ai-safety) | [IEEE AI Ethics Guidelines](https://ethicsinaction.ieee.org/)

---

## 知识库统计

| 维度 | 数据 |
|------|------|
| **章节数** | 8 个主要章节 |
| **文档总数** | 28 个内容文件 + 9 个索引文件 |
| **覆盖范围** | 从线性代数到 AI Agent，从理论到工程 |
| **每篇文档** | 300-500 行，含代码示例、对比表格、面试题 |

---
*Last updated: 2026-02-10*
