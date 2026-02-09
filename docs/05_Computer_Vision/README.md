# 05 计算机视觉 (Computer Vision)

本章涵盖图像理解与生成的核心技术，从经典 CNN 架构到目标检测（YOLO）、图像分割（Semantic/Instance）、多模态视觉（CLIP）以及生成模型（GAN/Diffusion）。这是视觉 AI 应用的技术全景。

## 学习路径 (Learning Path)

```
    ┌──────────────────────┐
    │  图像分类与检测       │
    │  Classification &    │
    │  Detection           │
    │  (ResNet/YOLO)       │
    └──────────┬───────────┘
               │
               ├────────────────────┐
               ▼                    ▼
    ┌──────────────────┐   ┌───────────────┐
    │  图像分割         │   │  多模态视觉   │
    │  Segmentation    │   │  Multimodal   │
    │  (U-Net/Mask)    │   │  (CLIP)       │
    └──────────────────┘   └───────────────┘
               │                    │
               └────────┬───────────┘
                        ▼
               ┌──────────────────┐
               │  生成模型         │
               │  Generative      │
               │  (GAN/Diffusion) │
               └──────────────────┘
```

## 内容索引 (Content Index)

| 主题 | 难度 | 描述 | 文档链接 |
|------|------|------|---------|
| 图像分类与检测 (Image Classification & Detection) | 入门 | CNN、ResNet、ViT、YOLO 系列，掌握图像识别基础 | [Image_Classification_Detection.md](./Image_Classification_Detection/Image_Classification_Detection.md) |
| 图像分割 (Segmentation) | 进阶 | 语义分割（U-Net）、实例分割（Mask R-CNN），像素级理解 | [Segmentation/](./Segmentation/) |
| 多模态视觉 (Multimodal Vision) | 进阶 | CLIP、ALIGN，视觉-语言联合表示学习 | [Multimodal_Vision/](./Multimodal_Vision/) |
| 生成模型 (Generative Models) | 实战 | GAN、DDPM、Stable Diffusion，图像生成与编辑 | [Generative_Models.md](./Generative_Models/Generative_Models.md) |

## 前置知识 (Prerequisites)

- **必修**: [神经网络核心](../03_Deep_Learning/Neural_Network_Core/Neural_Network_Core.md)（理解 CNN 架构）
- **必修**: [优化与正则化](../03_Deep_Learning/Optimization/Optimization.md)（训练视觉模型）
- **推荐**: [Transformer 革命](../04_NLP_LLMs/Transformer_Revolution/Transformer_Revolution.md)（理解 ViT 和多模态）
- **可选**: [概率统计](../01_Fundamentals/Probability_Statistics/Probability_Statistics.md)（理解扩散模型）

## 关键术语速查 (Key Terms)

- **卷积神经网络 (CNN)**: 利用局部感受野和权重共享处理图像的神经网络
- **ResNet (残差网络)**: 通过跳跃连接解决深层网络退化，CV 领域里程碑
- **ViT (Vision Transformer)**: 将图像分块用 Transformer 处理，打破 CNN 垄断
- **目标检测 (Object Detection)**: 定位并分类图像中多个对象（YOLO/Faster R-CNN）
- **语义分割 (Semantic Segmentation)**: 像素级分类，不区分实例（U-Net/DeepLab）
- **实例分割 (Instance Segmentation)**: 区分同类别不同实例（Mask R-CNN）
- **CLIP**: OpenAI 的视觉-语言预训练模型，实现零样本图像分类
- **GAN (生成对抗网络)**: 通过生成器-判别器对抗训练生成图像
- **Diffusion Model**: 通过逐步去噪生成图像，DALL-E/Stable Diffusion 核心
- **Latent Diffusion**: 在潜在空间执行扩散，大幅降低计算成本

---
*Last updated: 2026-02-10*
