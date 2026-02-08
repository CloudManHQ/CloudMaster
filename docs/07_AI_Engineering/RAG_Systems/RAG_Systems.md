# RAG 系统 (RAG Systems)

检索增强生成 (Retrieval-Augmented Generation) 通过结合外部实时知识解决 LLM 的幻觉问题。

## 1. 核心流程 (Core Pipeline)

### 索引 (Indexing)
- **文档分块 (Chunking)**: 将长文档切分为语义完整的片段。
- **向量化 (Embedding)**: 使用 OpenAI Ada 或开源的 BGE 模型。

### 检索 (Retrieval)
- **向量数据库**: Milvus, Pinecone, Weaviate。
- **混合搜索 (Hybrid Search)**: 结合关键词 (BM25) 与语义向量。
- **重排序 (Re-ranking)**: 使用交叉编码器 (Cross-Encoder) 提升结果精度。

### 生成 (Generation)
- **上下文拼接**: 将检索到的片段作为 Prompt 提供给 LLM。

## 2. 进阶技术
- **GraphRAG**: 利用知识图谱增强复杂关系的推理。
- **Self-RAG**: 模型自主判断是否需要检索及检索结果的可靠性。

## 3. 来源参考
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Pinecone: What is RAG?](https://www.pinecone.io/learn/retrieval-augmented-generation/)
