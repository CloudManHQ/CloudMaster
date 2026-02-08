# 模型部署与推理加速 (Deployment & Inference)

将 AI 模型高效转化为生产力的工程实践。

## 1. 推理引擎 (Inference Engines)

### vLLM
- **核心技术**: PagedAttention，通过分页管理显存极大提升吞吐量。
- **来源**: [vLLM: Easy, Fast, and Cheap LLM Serving](https://github.com/vllm-project/vllm)

### TensorRT (NVIDIA)
- **原理**: 针对 NVIDIA GPU 的层融合、内核自动调优与量化。

## 2. 量化与压缩 (Quantization & Compression)
- **Post-Training Quantization (PTQ)**: INT8/FP8 量化。
- **量化算法**: AWQ (Activation-aware Weight Quantization), GPTQ。
- **来源**: [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)

## 3. 服务架构
- **Triton Inference Server**: 支持多模型、多框架的统一推理平台。
