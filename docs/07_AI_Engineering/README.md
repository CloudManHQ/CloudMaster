# 07 AI 工程化与 MLOps (AI Engineering & MLOps)

本章聚焦将 AI 模型落地生产的工程实践，涵盖模型部署（推理加速/量化）、RAG 系统架构、MLOps 流水线（CI/CD/监控）和模型评估。这是 AI 从实验室走向产品的关键环节。

## 学习路径 (Learning Path)

```
    ┌──────────────────────┐
    │  模型评估             │
    │  Model Evaluation    │
    │  (指标/A/B测试)       │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  模型部署与推理       │
    │  Deployment &        │
    │  Inference           │
    │  (vLLM/量化)         │
    └──────────┬───────────┘
               │
               ├────────────────────┐
               ▼                    ▼
    ┌──────────────────┐   ┌───────────────┐
    │  RAG 系统         │   │  MLOps 流水线 │
    │  RAG Systems     │   │  MLOps        │
    │  (检索增强)       │   │  Pipeline     │
    └──────────────────┘   └───────────────┘
```

## 内容索引 (Content Index)

| 主题 | 难度 | 描述 | 文档链接 |
|------|------|------|---------|
| 模型部署与推理 (Deployment & Inference) | 进阶 | vLLM、TensorRT、量化技术（AWQ/GPTQ），优化推理性能 | [Deployment_Inference.md](./Deployment_Inference/Deployment_Inference.md) |
| RAG 系统 (RAG Systems) | 实战 | 向量数据库、混合检索、重排序，构建知识增强应用 | [RAG_Systems.md](./RAG_Systems/RAG_Systems.md) |
| MLOps 流水线 (MLOps Pipeline) | 实战 | 实验跟踪、模型注册、CI/CD、监控告警，自动化 ML 工作流 | [MLOps_Pipeline/](./MLOps_Pipeline/) |
| 模型评估 (Model Evaluation) | 进阶 | 离线指标、在线 A/B 测试、LLM 评估（MT-Bench/AlpacaEval） | [Model_Evaluation/](./Model_Evaluation/) |

## 前置知识 (Prerequisites)

- **必修**: [神经网络核心](../03_Deep_Learning/Neural_Network_Core/Neural_Network_Core.md)（理解模型结构）
- **必修**: [大语言模型架构](../04_NLP_LLMs/LLM_Architectures/LLM_Architectures.md)（部署 LLM）
- **推荐**: [分布式系统](../01_Fundamentals/Distributed_Systems/Distributed_Systems.md)（分布式推理）
- **推荐**: [Transformer 革命](../04_NLP_LLMs/Transformer_Revolution/Transformer_Revolution.md)（理解 RAG 中的编码器）

## 关键术语速查 (Key Terms)

- **推理加速 (Inference Optimization)**: 通过量化、剪枝、蒸馏提升模型推理速度
- **量化 (Quantization)**: 降低模型精度（FP16/INT8）减少显存和延迟
- **vLLM**: 高性能 LLM 推理引擎，支持连续批处理和 PagedAttention
- **TensorRT**: NVIDIA 推理优化库，深度优化 GPU 计算
- **RAG (Retrieval-Augmented Generation)**: 检索外部知识增强生成，缓解幻觉问题
- **向量数据库 (Vector Database)**: 存储和检索高维嵌入向量（Milvus/Qdrant）
- **重排序 (Reranking)**: 对初步检索结果精细排序，提升召回质量
- **MLOps**: 机器学习运维，覆盖训练、部署、监控全生命周期
- **Feature Store**: 特征存储系统，统一管理训练和推理特征
- **模型漂移 (Model Drift)**: 生产环境数据分布变化导致性能下降

---
*Last updated: 2026-02-10*
