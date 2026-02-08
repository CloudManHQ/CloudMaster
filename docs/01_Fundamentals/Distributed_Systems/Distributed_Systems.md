# 分布式系统 (Distributed Systems)

随着模型参数规模进入千亿、万亿级别，分布式系统成为 AI 工程的核心。

## 1. 核心挑战与架构 (Core Challenges & Architectures)

### 通信原语 (Communication Primitives)
- **All-Reduce**: 梯度同步的核心操作，将所有节点的梯度聚合后同步。
- **All-Gather**: 收集所有节点的数据。
- **Reduce-Scatter**: 分散聚合。
- **来源**: [NVIDIA Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl)

### 并行策略 (Parallelism Strategies)
- **数据并行 (Data Parallelism, DP)**: 每个节点复制模型，处理不同数据分片。
- **模型并行 (Model Parallelism, MP)**:
    - **张量并行 (Tensor Parallelism)**: 将单个层拆分到不同 GPU。
    - **流水线并行 (Pipeline Parallelism)**: 将模型的不同层分配到不同 GPU 阶段。
- **ZeRO (Zero Redundancy Optimizer)**: 消除冗余的模型状态存储。
- **来源**: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)

## 2. 计算设施与调度 (Infrastructure & Scheduling)

### 硬件架构
- **GPU 互联**: NVLink, NVSwitch。
- **网络拓扑**: Fat-Tree 架构、RoCEv2 协议。

### 集群管理
- **Kubernetes (K8s)**: 容器编排与任务调度。
- **Slurm**: 传统的 HPC 任务调度系统。

## 3. 推荐资源 (Recommended Resources)
- [Distributed Systems - Maarten van Steen](https://www.distributed-systems.net/index.php/books/ds3/)
- [Megatron-LM: Training Multi-Billion Parameter Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
