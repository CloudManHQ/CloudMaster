# AI 概念知识图谱

> AI 全链路概念之间的依赖关系与学习路径
> 
> 版本: 2026-04 | 概念数: 800+ | 关系类型: 依赖、前置、关联

---

## 📋 目录

1. [知识图谱总览](#一知识图谱总览)
2. [分层概念依赖](#二分层概念依赖)
3. [学习路径推荐](#三学习路径推荐)
4. [概念映射到文档](#四概念映射到文档)

---

## 一、知识图谱总览

### 1.1 概念层次结构

```
AI_Full_Stack
│
├── 1. Mathematical_Foundations
│   ├── Linear_Algebra
│   │   ├── Vector → Matrix → Tensor
│   │   ├── Dot_Product → Attention_Mechanism
│   │   └── Eigenvalue → PCA
│   │
│   ├── Probability_Statistics
│   │   ├── Bayesian_Theorem → Bayesian_Neural_Network
│   │   ├── KL_Divergence → VAE
│   │   ├── Entropy → Decision_Tree
│   │   ├── MLE/MAP → Parameter_Estimation
│   │   └── Sampling_Methods → MCMC/Gibbs
│   │
│   ├── Optimization
│   │   ├── Gradient_Descent → Backpropagation
│   │   ├── Adam/AdamW → Transformer_Training
│   │   ├── Convex_Optimization → SVM
│   │   ├── Learning_Rate_Schedule → Cosine/Warmup
│   │   └── Bayesian_Optimization → Hyperparameter_Tuning
│   │
│   ├── Calculus
│   │   ├── Chain_Rule → Backpropagation
│   │   ├── Partial_Derivative → Gradient
│   │   └── Taylor_Expansion → Approximation
│   │
│   ├── Information_Theory
│   │   ├── Cross_Entropy → Classification_Loss
│   │   └── Perplexity → Language_Model_Evaluation
│   │
│   └── Graph_Theory
│       ├── Graph → GNN
│       └── Adjacency_Matrix → Message_Passing
│
├── 2. Machine_Learning
│   ├── Supervised_Learning
│   │   ├── Linear_Regression → Neural_Network
│   │   ├── Logistic_Regression → Classification
│   │   ├── Decision_Tree → Random_Forest → XGBoost/LightGBM
│   │   ├── SVM → Kernel_Method
│   │   └── Ensemble_Methods → Bagging/Boosting/Stacking
│   │
│   ├── Unsupervised_Learning
│   │   ├── KMeans → Clustering
│   │   ├── PCA → Dimensionality_Reduction
│   │   ├── GMM → Mixture_Models
│   │   └── Anomaly_Detection → Isolation_Forest
│   │
│   ├── Transfer_Learning
│   │   ├── Domain_Adaptation → Distribution_Shift
│   │   ├── Pretrain_Finetune → Foundation_Model
│   │   └── Continual_Learning → Catastrophic_Forgetting
│   │
│   ├── Self_Supervised_Learning
│   │   ├── Contrastive_Learning → SimCLR/BYOL
│   │   ├── Masked_Modeling → BERT/MAE
│   │   └── JEPA → World_Model
│   │
│   ├── Meta_Learning
│   │   ├── Few_Shot → Prototypical_Network
│   │   ├── In_Context_Learning → LLM_Emergent
│   │   └── MAML → Task_Adaptation
│   │
│   ├── Feature_Engineering
│   │   ├── Feature_Selection → Filter/Wrapper/Embedded
│   │   ├── Feature_Crossing → Polynomial/Interaction
│   │   ├── Data_Cleaning → Missing_Value/Outlier
│   │   ├── EDA → Visualization/Distribution_Analysis
│   │   └── Auto_Feature_Engineering → Featuretools/TSFresh
│   │
│   ├── Federated_Learning
│   │   ├── FedAvg → Privacy_Preserving
│   │   ├── Differential_Privacy → Secure_Aggregation
│   │   ├── PPML → Privacy_Preserving_ML
│   │   ├── Homomorphic_Encryption → Encrypted_Computation
│   │   ├── Secure_MPC → Multi_Party_Computation
│   │   └── TEE → Trusted_Execution_Environment
│   │
│   ├── Reinforcement_Learning
│   │   ├── MDP → Bellman_Equation
│   │   ├── Q_Learning → DQN
│   │   ├── Policy_Gradient → REINFORCE → PPO
│   │   ├── Actor_Critic → A3C/SAC
│   │   ├── GRPO → LLM_RL
│   │   ├── Reward_Modeling → ORM/PRM
│   │   ├── RLHF → DPO/KTO/RLVR
│   │   └── Multi_Agent_RL → Cooperative/Competitive
│   │
│   └── Deep_Learning
│       ├── MLP → CNN → ResNet/EfficientNet
│       ├── RNN → LSTM/GRU → Transformer
│       ├── GAN → StyleGAN → Diffusion_Model
│       ├── Autoencoder → VAE → VQ-VAE
│       ├── State_Space_Model → S4 → Mamba/Mamba2
│       ├── GNN → GAT → Graph_Transformer
│       ├── Normalization → BatchNorm/LayerNorm/RMSNorm
│       ├── Activation → ReLU/GELU/SwiGLU
│       └── Regularization → Dropout/Weight_Decay
│
├── 3. NLP_Foundations
│   ├── Text_Processing
│   │   ├── Tokenization → BPE/WordPiece/SentencePiece
│   │   ├── Word_Embedding → Word2Vec/GloVe/FastText
│   │   └── Sentence_Embedding → Sentence_Transformers/E5
│   │
│   ├── Classic_NLP_Tasks
│   │   ├── Text_Classification → Sentiment_Analysis
│   │   ├── Named_Entity_Recognition → Sequence_Labeling
│   │   ├── Machine_Translation → Seq2Seq/Attention
│   │   ├── Question_Answering → Reading_Comprehension
│   │   └── Summarization → Abstractive/Extractive
│   │
│   └── Retrieval
│       ├── Bi_Encoder → Dense_Retrieval
│       ├── Cross_Encoder → Reranking
│       ├── Sparse_Retrieval → BM25/TF-IDF
│       └── Hybrid_Search → Dense+Sparse
│
├── 4. Large_Language_Models
│   ├── Architecture
│   │   ├── Transformer → BERT/GPT/T5
│   │   ├── Attention → MultiHead_Attention → GQA/MQA
│   │   ├── Positional_Encoding → RoPE/ALiBi
│   │   ├── Flash_Attention → Memory_Efficient
│   │   ├── Mixture_of_Experts → DeepSeek/Mixtral
│   │   ├── Mixture_of_Depths → Adaptive_Compute
│   │   └── State_Space_LLM → Mamba/Jamba
│   │
│   ├── Scaling
│   │   ├── Scaling_Laws → Chinchilla/Kaplan
│   │   ├── Test_Time_Compute → Inference_Scaling
│   │   ├── Reasoning_Models → o1/o3/DeepSeek_R1
│   │   ├── Long_Context → Ring_Attention/YaRN
│   │   └── Multi_Token_Prediction → Faster_Training
│   │
│   ├── Training
│   │   ├── Pretraining → MLM/Autoregressive/Prefix_LM
│   │   ├── Data_Engineering → Curation/Mixing/Dedup
│   │   ├── Curriculum_Learning → Data_Scheduling
│   │   ├── FineTuning → LoRA/QLoRA/DoRA
│   │   ├── Alignment → RLHF/DPO/KTO/RLVR
│   │   ├── GRPO → Group_Relative_Policy_Optimization
│   │   ├── Reward_Model → ORM/PRM/Verifier
│   │   ├── Knowledge_Distillation → Teacher_Student/On_Policy
│   │   └── Synthetic_Data → Self_Instruct/Evol_Instruct
│   │
│   ├── Inference
│   │   ├── Speculative_Decoding → Draft_Verify
│   │   ├── Structured_Output → Constrained_Decoding/JSON
│   │   ├── KV_Cache → Paged_Attention/Prefix_Caching
│   │   ├── Continuous_Batching → Dynamic_Scheduling
│   │   └── Sampling → Temperature/Top_p/Top_k/Min_p
│   │
│   ├── Evaluation_Benchmarks
│   │   ├── Knowledge → MMLU/MMLU-Pro/GPQA
│   │   ├── Coding → HumanEval/MBPP/SWE-bench
│   │   ├── Reasoning → GSM8K/MATH/ARC
│   │   ├── Dialogue → MT-Bench/Chatbot_Arena/AlpacaEval
│   │   ├── Truthfulness → TruthfulQA/SimpleQA
│   │   ├── Safety → HarmBench/ToxiGen
│   │   ├── Multimodal → MMMU/MMBench
│   │   ├── Chinese → C-Eval/CMMLU
│   │   └── Meta_Eval → LLM_as_Judge/OpenCompass/lm-eval-harness
│   │
│   └── Application
│       ├── Prompt_Engineering → CoT/ToT/ReAct
│       ├── RAG → Vector_Database
│       ├── Agent → Tool_Use
│       ├── Code_Generation → Copilot/Codex
│       └── Guardrails → Input/Output_Filtering
│
├── 5. Computer_Vision
│   ├── Image_Classification
│   │   ├── CNN → ResNet/EfficientNet
│   │   ├── ViT → DeiT/Swin_Transformer
│   │   └── Foundation_Model → DINOv2/SigLIP
│   │
│   ├── Object_Detection
│   │   ├── YOLO → YOLOv8/YOLO-World
│   │   ├── DETR → End_to_End_Detection
│   │   └── Open_Vocabulary → Grounding_DINO/OWL-ViT
│   │
│   ├── Segmentation
│   │   ├── Semantic → FCN/DeepLab
│   │   ├── Instance → Mask_R-CNN
│   │   ├── Panoptic → Unified_Segmentation
│   │   └── SAM → SAM2/Segment_Anything
│   │
│   └── 3D_Vision
│       ├── NeRF → Neural_Radiance_Field
│       ├── 3D_Gaussian_Splatting → Real_Time_Rendering
│       ├── Depth_Estimation → Monocular/Stereo
│       └── Point_Cloud → PointNet/PointNet++
│
├── 6. Multimodal_Models
│   ├── Vision_Language
│   │   ├── CLIP → Contrastive_Learning
│   │   ├── ViT → Vision_Encoder
│   │   ├── LLaVA → Instruction_Tuning
│   │   └── Qwen-VL/InternVL → Native_Multimodal
│   │
│   ├── Video
│   │   ├── Temporal_Modeling → V-JEPA
│   │   ├── Video_LLM → Native_Video
│   │   └── Video_Generation → Sora/Kling/Runway
│   │
│   ├── Speech_Audio
│   │   ├── ASR → Whisper/Conformer
│   │   ├── TTS → VALL-E/F5-TTS/CosyVoice
│   │   └── Audio_LLM → Qwen-Audio/SALMONN
│   │
│   ├── 3D_Generation
│   │   ├── NeRF → 3D_Reconstruction
│   │   ├── 3DGS → Gaussian_Splatting_Generation
│   │   └── Point-E/Shap-E → 3D_Assets
│   │
│   ├── Generation
│   │   ├── Diffusion_Model → Stable_Diffusion/FLUX
│   │   ├── Autoregressive → GPT4o/Gemini
│   │   ├── Text_to_Image → DALL-E/Midjourney
│   │   └── Flow_Matching → Rectified_Flow
│   │
│   └── Cross_Modal
│       ├── Contrastive_Learning → CLIP/SigLIP
│       ├── Cross_Attention → Fusion_Strategy
│       └── Omni_Modal → Any_to_Any
│
├── 7. AI_Agents
│   ├── Foundation
│   │   ├── LLM → Agent_Brain
│   │   ├── Tool_Use → Function_Calling
│   │   ├── Memory → Short/Long_Term/Episodic
│   │   ├── Perception → Multimodal_Input
│   │   └── Grounding → Environment_Interaction
│   │
│   ├── Architecture
│   │   ├── ReAct → Reasoning_Acting
│   │   ├── Reflexion → Self_Correction
│   │   ├── MultiAgent → Collaboration/Debate
│   │   ├── Planning → Task_Decomposition/Hierarchical
│   │   └── Human_in_the_Loop → Oversight/Approval
│   │
│   ├── Protocols
│   │   ├── MCP → Tool_Standardization
│   │   ├── A2A → Agent_Communication
│   │   ├── AG-UI → User_Interface_Streaming
│   │   └── UCP → Resource_Scheduling
│   │
│   ├── Frameworks
│   │   ├── LangChain/LangGraph → Workflow_Orchestration
│   │   ├── CrewAI → Role_Based_MultiAgent
│   │   ├── AutoGen → Conversational_Agent
│   │   └── OpenAI_Agents_SDK → Function_Calling
│   │
│   ├── Patterns
│   │   ├── Coding_Agent → Devin/Cursor/Windsurf
│   │   ├── Computer_Use → Browser/Desktop_Agent
│   │   ├── Agentic_RAG → Retrieve_Reason_Act
│   │   ├── Agentic_Loops → Iterative_Refinement
│   │   ├── Workflow_Graphs → DAG_Orchestration
│   │   └── Context_Management → Window/Compression
│   │
│   ├── Security
│   │   ├── Sandboxing → Isolation/Permission
│   │   ├── Agent_Guardrails → Scope_Limiting
│   │   └── Observability → Trace/Log/Monitor
│   │
│   └── Evaluation
│       ├── Agent_Harness → Benchmark
│       ├── GAIA → General_Capability
│       ├── SWE-bench → Coding_Evaluation
│       └── Safety → Red_Teaming
│
├── 8. RAG_Systems
│   ├── RAG_Paradigms
│   │   ├── Naive_RAG → Retrieve_Generate
│   │   ├── Advanced_RAG → Query_Rewrite/HyDE
│   │   ├── Graph_RAG → Knowledge_Graph_Retrieval
│   │   ├── Agentic_RAG → Multi_Step_Retrieval
│   │   └── Multimodal_RAG → Image/Table_Retrieval
│   │
│   ├── Components
│   │   ├── Chunking → Fixed/Semantic/Recursive
│   │   ├── Embedding_Model → BGE/E5/OpenAI
│   │   ├── Vector_Database → Milvus/Pinecone/Qdrant
│   │   ├── Reranking → Cross_Encoder/ColBERT
│   │   └── Hybrid_Search → Dense+Sparse+Keyword
│   │
│   └── Evaluation
│       ├── Faithfulness → Hallucination_Detection
│       ├── Relevance → RAGAS/TruLens
│       └── Context_Quality → Precision/Recall
│
├── 9. Embodied_AI
│   ├── VLA
│   │   ├── Vision_Encoder → ViT/SigLIP
│   │   ├── Language_Model → LLM
│   │   ├── Action_Decoder → Flow_Matching/Diffusion
│   │   └── Models → π0/RT-2/Octo/RDT
│   │
│   ├── Learning_Methods
│   │   ├── Imitation_Learning → Behavior_Cloning/DAgger
│   │   ├── RL_for_Robotics → Sim_Training
│   │   └── Curriculum_Learning → Task_Progression
│   │
│   ├── Robotics
│   │   ├── Manipulation → Grasping/Dexterous
│   │   ├── Navigation → SLAM/Visual_Nav
│   │   ├── Humanoid → Whole_Body_Control
│   │   └── Locomotion → Legged/Wheeled
│   │
│   ├── Sim2Real
│   │   ├── Domain_Randomization → Robustness
│   │   ├── Digital_Twin → Environment_Replica
│   │   ├── Simulation → IsaacGym/MuJoCo
│   │   └── World_Model → JEPA/Video_Prediction
│   │
│   └── Perception
│       ├── Open_Vocabulary → Language_Grounding
│       ├── 3D_Understanding → Point_Cloud/Depth
│       └── Tactile_Sensing → Force/Contact
│
├── 10. AI_Infrastructure
│   ├── Hardware
│   │   ├── GPU → H100/B200/GB300
│   │   ├── TPU → v5p/v6e
│   │   ├── Edge → Jetson/Apple_Neural_Engine
│   │   └── Interconnect → NVLink/InfiniBand
│   │
│   ├── Training_Systems
│   │   ├── Data_Parallel → FSDP/Distributed
│   │   ├── Model_Parallel → Pipeline/Tensor/Expert
│   │   ├── ZeRO → Memory_Optimization
│   │   ├── DeepSpeed → Megatron-LM
│   │   ├── Flash_Attention → IO_Aware_Attention
│   │   └── Checkpointing → Gradient/Activation
│   │
│   ├── Inference_Systems
│   │   ├── vLLM → PagedAttention
│   │   ├── SGLang → Structured_Output
│   │   ├── TensorRT-LLM → NVIDIA_Inference
│   │   ├── Quantization → GPTQ/AWQ/GGUF
│   │   ├── Pruning → Structured/Unstructured
│   │   └── Serving → Triton/Ray_Serve
│   │
│   ├── Data_Infrastructure
│   │   ├── Feature_Store → Feast/Tecton
│   │   ├── Data_Pipeline → ETL/Streaming
│   │   ├── Data_Quality → Validation/Monitoring
│   │   └── Annotation → Label_Studio/Scale
│   │
│   ├── MLOps
│   │   ├── Kubeflow → Pipeline_Orchestration
│   │   ├── MLflow → Experiment_Tracking
│   │   ├── Model_Registry → Versioning/Staging
│   │   ├── Monitoring → Drift_Detection/Alerting
│   │   └── A_B_Testing → Canary/Shadow_Deploy
│   │
│   └── LLMOps
│       ├── Prompt_Management → Version/Template
│       ├── Eval_Pipeline → Auto_Eval/Human_Eval
│       ├── Gateway → Rate_Limit/Routing/Fallback
│       └── Cost_Optimization → Caching/Model_Routing
│
├── 12. AI_for_Science
│   ├── Life_Science
│   │   ├── AlphaFold → Protein_Structure_Prediction
│   │   ├── Drug_Discovery → Molecular_Generation/Virtual_Screening
│   │   ├── Genomics → DNA_Language_Model/Variant_Calling
│   │   └── Medical_Imaging → Pathology_AI/Radiology_AI
│   │
│   ├── Physical_Science
│   │   ├── Weather_Prediction → Pangu-Weather/GraphCast/GenCast
│   │   ├── Materials_Discovery → GNoME/Crystal_Generation
│   │   ├── Physics_Simulation → Neural_PDE/PINN
│   │   └── Molecular_Dynamics → Force_Field/Ab_Initio
│   │
│   ├── Mathematics
│   │   ├── Theorem_Proving → Lean/Coq/AlphaProof
│   │   ├── Symbolic_Regression → Equation_Discovery
│   │   └── Math_Reasoning → DeepSeek-Prover/Minerva
│   │
│   └── Earth_Science
│       ├── Climate_Modeling → Earth_Digital_Twin
│       ├── Remote_Sensing → Satellite_Analysis
│       └── Geological_Survey → Mineral_Exploration
│
└── 13. Safety_Ethics
    ├── Alignment
    │   ├── RLHF → Human_Preference
    │   ├── Constitutional_AI → Principles
    │   └── Red_Teaming → Safety_Evaluation
    │
    ├── Interpretability
    │   ├── Mechanistic_Interpretability → Circuit_Analysis
    │   ├── Feature_Visualization → Activation_Patching
    │   └── Explainable_AI → SHAP/LIME
    │
    ├── Security
    │   ├── Prompt_Injection → Jailbreaking
    │   ├── AI_Watermarking → Content_Authentication
    │   ├── Adversarial_Attack → Robustness
    │   └── Frontier_Model_Risk → Capability_Evaluation
    │
    └── Governance
        ├── EU_AI_Act → Regulation
        ├── Model_Card → Transparency
        ├── Bias → Fairness
        └── Open_Source_Licensing → Model_Release
```

---

## 二、分层概念依赖

### 2.1 核心依赖链

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    核心概念依赖链                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  链 1: 从数学到 Transformer                                             │
│  ───────────────────────────                                            │
│  Linear_Algebra → Gradient_Descent → Backpropagation → MLP → CNN →     │
│  ResNet → Attention → Transformer → BERT/GPT → ChatGPT                 │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  链 2: 从 Transformer 到 Agent                                          │
│  ──────────────────────────────                                         │
│  Transformer → GPT → Instruction_Tuning → RLHF → Tool_Use → ReAct →    │
│  Agent → MCP/A2A → MultiAgent                                          │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  链 3: 从 Vision 到 VLA                                                 │
│  ───────────────────────                                                │
│  CNN → ViT → Vision_Transformer → CLIP → Contrastive_Learning →        │
│  Vision_Language_Model → VLA → π0                                      │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  链 4: 从 Training 到 Production                                        │
│  ─────────────────────────────                                          │
│  Training → Checkpoint → Quantization → ONNX → TensorRT →              │
│  vLLM → Production                                                     │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  链 5: 从 Scaling Laws 到 Reasoning                                      │
│  ────────────────────────────────                                       │
│  Scaling_Laws → Chinchilla → MoE → Test_Time_Compute →                 │
│  Reasoning_Models → o1/o3 → Agent_Reasoning                            │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  链 6: 从 Interpretability 到 Governance                                 │
│  ──────────────────────────────────────                                 │
│  Mechanistic_Interpretability → Feature_Circuit → Safety_Evaluation →  │
│  Red_Teaming → Frontier_Risk → EU_AI_Act → Governance                  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  链 7: 从 Tokenization 到 RAG                                            │
│  ─────────────────────────────                                          │
│  Tokenization → Embedding → Bi_Encoder → Vector_DB → Chunking →       │
│  Hybrid_Search → Reranking → Naive_RAG → Advanced_RAG → Agentic_RAG   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  链 8: 从 RL 到 LLM Alignment                                           │
│  ─────────────────────────────                                          │
│  MDP → Q_Learning → Policy_Gradient → PPO → RLHF → DPO →              │
│  GRPO → RLVR → Reward_Model → PRM → Reasoning_RL                      │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  链 9: 从 CNN 到 3D Generation                                           │
│  ──────────────────────────────                                         │
│  CNN → Object_Detection → Segmentation → SAM → NeRF →                  │
│  3D_Gaussian_Splatting → Diffusion_3D → World_Simulator                │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  链 10: 从 Agent 到 Computer Use                                         │
│  ─────────────────────────────                                          │
│  LLM → Tool_Use → Function_Calling → MCP → Agent →                     │
│  Computer_Use → Browser_Agent → Coding_Agent → Agentic_IDE             │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  链 11: 从 AI 到 Scientific Discovery                                    │
│  ───────────────────────────────────                                    │
│  DL → GNN → AlphaFold → Protein_Prediction →                           │
│  Drug_Discovery → Molecular_Generation → Clinical_AI                    │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  链 12: 从 Data 到 Feature 到 Model                                      │
│  ────────────────────────────────                                       │
│  Raw_Data → EDA → Data_Cleaning → Feature_Engineering →                │
│  Feature_Selection → Model_Training → Evaluation → Production          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 概念关联矩阵

| 概念 | 强关联 | 弱关联 |
|------|--------|--------|
| **Transformer** | Attention, BERT, GPT, LLM | CNN, RNN |
| **LLM** | Prompt, RLHF, Agent, RAG | Traditional_NLP |
| **Agent** | Tool_Use, MCP, ReAct, Planning | Chatbot |
| **VLA** | ViT, LLM, Flow_Matching, Robotics | SLAM |
| **JEPA** | Self_Supervised, World_Model, Video | Contrastive_Learning |
| **RAG** | Vector_DB, Embedding, LLM | Fine_tuning |
| **MCP** | Tool, API, Agent, Protocol | Function_Calling |
| **MoE** | Sparse_Activation, Router, Expert | Dense_Model |
| **Scaling_Laws** | Compute, Data, Parameters | Heuristic_Tuning |
| **Test_Time_Compute** | Reasoning, Search, Verification | Standard_Inference |
| **Mamba/SSM** | State_Space, Linear_Attention, Sequence | Transformer |
| **GRPO** | PPO, RLHF, DeepSeek_R1, Reward | DPO |
| **RLVR** | Verifier, Math_Reasoning, Code, RL | SFT |
| **Flash_Attention** | IO_Aware, Training, Inference, Memory | Standard_Attention |
| **Graph_RAG** | Knowledge_Graph, Entity, Relationship | Naive_RAG |
| **SAM** | Segmentation, Foundation_Model, Zero_Shot | Mask_R-CNN |
| **3DGS** | Gaussian_Splatting, NeRF, Real_Time | Point_Cloud |
| **Coding_Agent** | Code_Gen, Tool_Use, IDE, SWE-bench | Copilot |
| **Digital_Twin** | Simulation, Sim2Real, Robotics | Physical_Model |
| **Imitation_Learning** | Behavior_Cloning, Demo, VLA | RL |
| **AlphaFold** | Protein, GNN, Structure, Biology | Drug_Discovery |
| **MMLU** | Knowledge, Benchmark, Multi-task | HumanEval |
| **Feature_Engineering** | Selection, Crossing, EDA, Pipeline | Raw_Data |
| **PPML** | Federated, Encryption, Privacy | Differential_Privacy |

---

## 三、学习路径推荐

### 3.1 不同角色的学习路径

#### 路径 1: AI 研究员 (Researcher)

```
阶段 1: 基础 (3-6 个月)
├── Linear Algebra → Probability → Optimization
├── Machine Learning → Deep Learning
└── PyTorch/TensorFlow

阶段 2: 专精 (6-12 个月)
├── Transformer Architecture (Attention is All You Need)
├── BERT/GPT Paper Reading
├── Training at Scale (DeepSpeed, Megatron)
└── Experiment Design

阶段 3: 前沿 (持续)
├── Multimodal Models
├── Agent Systems
├── World Models (JEPA)
└── Safety & Alignment
```

#### 路径 2: AI 工程师 (Engineer)

```
阶段 1: 基础 (2-3 个月)
├── Python → PyTorch
├── ML Basics → NLP Foundations
├── Tokenization → Embedding
└── Software Engineering

阶段 2: LLM 工程 (3-6 个月)
├── Prompt Engineering (CoT, Few-Shot)
├── Fine-tuning (LoRA, QLoRA, DoRA)
├── RAG Systems (Naive → Advanced → Graph → Agentic)
├── LLM Deployment (vLLM, Quantization)
├── Structured Output & Guardrails
└── MCP/A2A/AG-UI Integration

阶段 3: Agent 工程 (3-6 个月)
├── LangChain/LangGraph/CrewAI
├── Agent Design Patterns (ReAct, Reflexion)
├── Coding Agent / Computer Use
├── Tool Development (MCP Server)
├── Production Deployment & LLMOps
└── Observability & Cost Optimization
```

#### 路径 3: 机器人工程师 (Robotics Engineer)

```
阶段 1: 基础 (3-6 个月)
├── ROS/ROS2
├── Kinematics/Dynamics
├── Computer Vision
└── Control Theory

阶段 2: 学习 (6-12 个月)
├── Imitation Learning
├── Reinforcement Learning
├── VLA Models (π0, RDT)
├── Sim2Real
└── ROS-LLM Integration

阶段 3: 集成 (6-12 个月)
├── Multi-modal Perception
├── Whole-body Control
├── Human-Robot Interaction
└── Production Deployment
```

#### 路径 4: AI 安全研究员 (Safety Researcher)

```
阶段 1: 基础 (3-6 个月)
├── Deep Learning → Training Dynamics
├── NLP → LLM Architecture
├── RL Basics → RLHF
└── Statistics → Evaluation Methods

阶段 2: 核心 (6-12 个月)
├── Mechanistic Interpretability
├── Red Teaming Techniques
├── Adversarial Attack / Prompt Injection
├── Alignment (RLHF/DPO/Constitutional AI)
└── Reward Modeling (ORM/PRM)

阶段 3: 前沿 (持续)
├── AI Governance & Policy
├── Frontier Model Evaluation
├── AI Watermarking
├── Scalable Oversight
└── Agent Safety & Sandboxing
```

#### 路径 5: 数据科学家 (Data Scientist)

```
阶段 1: 基础 (2-3 个月)
├── Statistics → Probability → Hypothesis Testing
├── Python → Pandas/NumPy/Matplotlib
└── SQL → Data Querying

阶段 2: 机器学习 (3-6 个月)
├── EDA → Data Visualization
├── Feature Engineering (Selection, Crossing)
├── Supervised Learning (XGBoost, LightGBM)
├── Unsupervised Learning (Clustering, PCA)
└── Model Evaluation (Cross-Validation, A/B Testing)

阶段 3: 进阶 (3-6 个月)
├── Deep Learning → PyTorch
├── NLP / LLM 基础
├── Experiment Tracking (MLflow)
├── Feature Store (Feast)
└── Production ML Pipeline
```

#### 路径 6: RAG/检索工程师 (Retrieval Engineer)

```
阶段 1: 基础 (2-3 个月)
├── NLP → Tokenization → Embedding
├── Information Retrieval (BM25, TF-IDF)
├── Sentence Transformers / Bi-Encoder
└── Vector Database (Milvus, Qdrant)

阶段 2: RAG 系统 (3-6 个月)
├── Naive RAG → Advanced RAG
├── Chunking Strategies
├── Hybrid Search (Dense + Sparse)
├── Reranking (Cross-Encoder, ColBERT)
├── Graph RAG → Knowledge Graph
└── Multimodal RAG

阶段 3: 生产化 (3-6 个月)
├── Agentic RAG
├── Evaluation (RAGAS, Faithfulness)
├── Guardrails & Hallucination Detection
├── Production Optimization
└── LLMOps & Monitoring
```

### 3.2 概念学习顺序

```
必须先学 → 再学 → 最后学

Example 1: 理解 VLA
Linear_Algebra → CNN → ViT → CLIP → 
Vision_Language_Model → VLA → π0

Example 2: 部署 LLM
Transformer → GPT → Fine_tuning → 
Quantization → ONNX → TensorRT → vLLM

Example 3: 构建 Agent
LLM → Tool_Use → ReAct → MCP → 
LangChain → MultiAgent → Computer_Use

Example 4: 理解 Reasoning Models
Scaling_Laws → Test_Time_Compute → 
CoT → Tree_Search → GRPO → PRM → o1/o3

Example 5: 理解 MoE 架构
Transformer → Sparse_Activation → Router → 
Expert_Parallel → DeepSeek_V3 → Mixtral

Example 6: AI 安全研究
Neural_Network → Interpretability → 
Mechanistic_Interpretability → Circuit_Analysis → 
Red_Teaming → Alignment

Example 7: 构建 RAG 系统
Tokenization → Embedding → Vector_DB → 
Chunking → Hybrid_Search → Reranking → 
Naive_RAG → Advanced_RAG → Graph_RAG → Agentic_RAG

Example 8: LLM Alignment
RL_Basics → Policy_Gradient → PPO → 
RLHF → DPO → GRPO → RLVR → PRM/ORM

Example 9: CV 到 3D 生成
CNN → Object_Detection → Segmentation → SAM →
NeRF → 3D_Gaussian_Splatting → Diffusion_3D

Example 10: Coding Agent
LLM → Code_Generation → Tool_Use → MCP →
Sandboxing → SWE-bench → Coding_Agent → Agentic_IDE

Example 11: AI for Science
DL → GNN → Protein_Representation → AlphaFold →
Drug_Discovery → Molecular_Generation → Clinical_AI

Example 12: LLM Evaluation
LLM → MMLU → HumanEval → MT-Bench →
LLM_as_Judge → OpenCompass → Chatbot_Arena

Example 13: 数据科学 Pipeline
Raw_Data → EDA → Data_Cleaning →
Feature_Engineering → Feature_Selection →
Model_Training → Evaluation → Deployment
```

---

## 四、概念映射到文档

### 4.1 文档索引

| 概念类别 | 主要文档 | 相关文档 |
|----------|----------|----------|
| **数学基础** | `01_Fundamentals/` | `Linear_Algebra/`, `Probability_Statistics/` |
| **机器学习** | `02_Machine_Learning/` | `Feature_Engineering/`, `Supervised_Learning/`, `Unsupervised_Learning/` |
| **深度学习** | `03_Deep_Learning/` | `Neural_Network_Core/`, `Optimization/`, `World_Models/JEPA_Architecture_2026.md` |
| **NLP 基础** | `04_NLP_LLMs/Sequence_Models/` | `Transformer_Revolution/` |
| **大模型** | `04_NLP_LLMs/LLM_Architectures/` | `Fine_tuning_Techniques/`, `Prompt_Engineering/` |
| **多模态** | `04_NLP_LLMs/Multimodal_Models/` | `05_Computer_Vision/` |
| **计算机视觉** | `05_Computer_Vision/` | `Image_Classification_Detection/`, `Segmentation/`, `Generative_Models/` |
| **强化学习** | `06_Reinforcement_Learning/` | `RL_Foundations/`, `Deep_RL/` |
| **Agent** | `06_Reinforcement_Learning/AI_Agents/` | `Agent_Protocols_Detail.md`, `Agent_Future_Roadmap_2026_2030.md` |
| **RAG 系统** | `07_AI_Engineering/RAG_Systems/` | `RAG_Advanced_2026/` |
| **MoE/Scaling** | `04_NLP_LLMs/LLM_Architectures/` | `AI_Infrastructure_Trends_2026.md` |
| **Reasoning** | `04_NLP_LLMs/LLM_Architectures/` | `Prompt_Engineering.md` |
| **AI for Science** | `13_AI_Applications_Industry/` | `AI_Applications_Industry.md` |
| **评估基准** | `07_AI_Engineering/Model_Evaluation/` | `12_Agent_Evaluation/`, `Benchmarking/` |
| **特征工程** | `02_Machine_Learning/Feature_Engineering/` | `Feature_Engineering.md` |
| **安全/可解释** | `08_Ethics_Safety/` | `AI_Safety_RedTeaming/`, `AI_Security_2026/`, `Value_Alignment/` |
| **具身智能/VLA** | `06_Reinforcement_Learning/Robotics_Embodied_AI/` | `VLA_Models_2026.md`, `Embodied_AI_Complete_2026.md` |
| **评估** | `12_Agent_Evaluation/` | `Agent_Harness_Complete_2026.md`, `Benchmarking/`, `Metrics/` |
| **基础设施** | `07_AI_Engineering/` | `Deployment_Inference/`, `MLOps_Pipeline/`, `AI_Infrastructure_Trends_2026.md` |
| **LLMOps** | `07_AI_Engineering/` | `Model_Evaluation/`, `AI_Coding_Assistants/` |

### 4.2 快速查找

```
想找什么？

Q: MCP 协议详解
A: 06_Reinforcement_Learning/AI_Agents/Agent_Protocols_Detail.md

Q: 多模态模型架构
A: 04_NLP_LLMs/Multimodal_Models/Multimodal_Architectures_2026.md

Q: VLA 模型技术
A: 06_Reinforcement_Learning/Robotics_Embodied_AI/VLA_Models_2026.md

Q: JEPA 世界模型
A: 03_Deep_Learning/World_Models/JEPA_Architecture_2026.md

Q: 具身智能完整指南
A: 06_Reinforcement_Learning/Robotics_Embodied_AI/Embodied_AI_Complete_2026.md

Q: Agent 评估框架
A: 12_Agent_Evaluation/Agent_Harness_Complete_2026.md

Q: Agent 未来发展
A: 06_Reinforcement_Learning/AI_Agents/Agent_Future_Roadmap_2026_2030.md

Q: AI 基础设施趋势
A: 07_AI_Engineering/AI_Infrastructure_Trends_2026.md

Q: RAG 系统设计
A: 07_AI_Engineering/RAG_Systems/

Q: 强化学习基础
A: 06_Reinforcement_Learning/RL_Foundations/

Q: 计算机视觉目标检测
A: 05_Computer_Vision/Image_Classification_Detection/

Q: 图像分割 (SAM)
A: 05_Computer_Vision/Segmentation/

Q: Coding Agent / AI 编码助手
A: 07_AI_Engineering/AI_Coding_Assistants/

Q: LLM 微调 (LoRA/QLoRA)
A: 04_NLP_LLMs/Fine_tuning_Techniques/

Q: 视频生成
A: 05_Computer_Vision/Video_Generation/Video_Generation_2026.md

Q: 分布式训练
A: 01_Fundamentals/Distributed_Systems/

Q: AI 对齐与价值观
A: 08_Ethics_Safety/Value_Alignment/

Q: LLM 评估基准 (MMLU/HumanEval)
A: 07_AI_Engineering/Model_Evaluation/Model_Evaluation.md

Q: 特征工程
A: 02_Machine_Learning/Feature_Engineering/Feature_Engineering.md

Q: AI for Science (药物研发/蛋白质)
A: 13_AI_Applications_Industry/AI_Applications_Industry.md

Q: 隐私计算/联邦学习
A: 02_Machine_Learning/ (Federated_Learning 相关)

Q: Agent 评估框架 (RAPS/基准测试)
A: 12_Agent_Evaluation/
```

---

## 附录：概念速查表

### A. 2026 热门概念速查

| 概念 | 定义 | 相关技术 | 文档 |
|------|------|----------|------|
| **MCP** | 模型上下文协议 | Agent, Tool | Agent_Protocols_Detail.md |
| **VLA** | 视觉-语言-动作 | Robotics, π0 | VLA_Models_2026.md |
| **JEPA** | 联合嵌入预测架构 | World Model | JEPA_Architecture_2026.md |
| **Flow Matching** | 流匹配生成 | Diffusion, π0 | VLA_Models_2026.md |
| **SGLang** | 结构化生成语言 | Inference | AI_Infrastructure_Trends_2026.md |
| **Agent Harness** | Agent 评估框架 | Evaluation | Agent_Harness_Complete_2026.md |
| **MoE** | 混合专家模型 | DeepSeek, Mixtral | LLM_Architectures.md |
| **Test-Time Compute** | 推理时计算扩展 | o1, Reasoning | LLM_Architectures.md |
| **Mamba/SSM** | 状态空间模型 | Linear Attention | Neural_Network_Core.md |
| **AG-UI** | Agent 用户界面协议 | Agent, UI | Agent_Protocols_Detail.md |
| **Mechanistic Interp.** | 机制可解释性 | Safety, Circuit | AI_Security_2026/ |
| **Speculative Decoding** | 推测解码 | Inference, Speed | AI_Infrastructure_Trends_2026.md |
| **GRPO** | 组相对策略优化 | DeepSeek-R1, RL | LLM_Architectures.md |
| **RLVR** | 可验证奖励强化学习 | Math, Code, Verifier | LLM_Architectures.md |
| **Graph RAG** | 图增强检索生成 | Knowledge Graph | RAG_Advanced_2026/ |
| **Agentic RAG** | 智能体驱动 RAG | Multi-step, Agent | RAG_Advanced_2026/ |
| **SAM/SAM2** | 分割一切模型 | Zero-shot, CV | Segmentation/ |
| **3D Gaussian Splatting** | 3D 高斯泼溅 | NeRF, Rendering | Video_Generation_2026.md |
| **Coding Agent** | 编码智能体 | Devin, SWE-bench | AI_Coding_Assistants/ |
| **Computer Use** | 计算机使用代理 | Browser, Desktop | AI_Agents/ |
| **Digital Twin** | 数字孪生 | Sim2Real, Robotics | Embodied_AI_Complete_2026.md |
| **Flash Attention** | 闪存注意力 | IO-Aware, GPU | AI_Infrastructure_Trends_2026.md |
| **Continuous Batching** | 连续批处理 | Inference, vLLM | Deployment_Inference/ |
| **LLMOps** | 大模型运维 | Eval, Gateway | AI_Infrastructure_Trends_2026.md |
| **MMLU** | 多任务知识评测 | 57学科, Benchmark | Model_Evaluation.md |
| **HumanEval** | 代码生成评测 | pass@k, Coding | Model_Evaluation.md |
| **MT-Bench** | 多轮对话评测 | LLM-as-Judge | Model_Evaluation.md |
| **AlphaFold** | 蛋白质结构预测 | GNN, Biology | AI_Applications_Industry.md |
| **GraphCast** | AI 天气预报 | Weather, GNN | AI_Applications_Industry.md |
| **AlphaProof** | AI 数学证明 | Lean, Theorem | AI_Applications_Industry.md |

### B. 经典基础概念速查

| 概念 | 定义 | 相关技术 | 文档 |
|------|------|----------|------|
| **Tokenization** | 文本分词 | BPE, SentencePiece | Sequence_Models/ |
| **Embedding** | 向量表示 | Word2Vec, BERT | Sequence_Models/ |
| **Attention** | 注意力机制 | Self/Cross/Multi-Head | Transformer_Revolution/ |
| **Backpropagation** | 反向传播 | Chain Rule, Gradient | Neural_Network_Core/ |
| **CNN** | 卷积神经网络 | ResNet, EfficientNet | Neural_Network_Core/ |
| **RNN/LSTM** | 循环神经网络 | Sequence Modeling | Sequence_Models/ |
| **GAN** | 生成对抗网络 | StyleGAN, Image Gen | Generative_Models/ |
| **RL (MDP)** | 强化学习基础 | Q-Learning, Policy | RL_Foundations/ |
| **PPO** | 近端策略优化 | Actor-Critic, RLHF | Deep_RL/ |
| **Transfer Learning** | 迁移学习 | Pretrain-Finetune | Supervised_Learning/ |
| **Ensemble Methods** | 集成方法 | Bagging, Boosting | Supervised_Learning/ |
| **Object Detection** | 目标检测 | YOLO, DETR | Image_Classification_Detection/ |
| **Segmentation** | 图像分割 | Semantic, Instance | Segmentation/ |
| **Feature Engineering** | 特征工程 | Selection, EDA | Feature_Engineering/ |
| **Federated Learning** | 联邦学习 | FedAvg, Privacy | Distributed_Systems/ |
| **PPML** | 隐私保护ML | HE, MPC, TEE | AI_Security_2026/ |

---

*Last updated: 2026-04-03*
*全面更新: 从 8 大模块扩展至 13 大模块 (新增 NLP_Foundations、Computer_Vision、RAG_Systems、AI_for_Science)，新增 200+ 概念，12 条依赖链，6 条学习路径，覆盖 RL 基础、Tokenization/Embedding、GRPO/RLVR、Graph RAG、SAM、3DGS、Coding Agent、LLMOps、Evaluation Benchmarks、Feature Engineering、Privacy Computing、AI for Science 等全部关键领域*
