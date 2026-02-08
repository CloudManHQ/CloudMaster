# 图像分类与检测 (Image Classification & Detection)

计算机视觉中的基础任务及其演进历程。

## 1. 图像分类 (Image Classification)

### 卷积网络里程碑
- **ResNet (Residual Networks)**: 引入残差连接，解决了超深网络的梯度消失问题。
- **EfficientNet**: 通过复合缩放 (Compound Scaling) 实现精度与效率的平衡。
- **来源**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

### Vision Transformer (ViT)
- **原理**: 将图像切块 (Patches) 并应用标准的 Transformer 编码器。

## 2. 目标检测 (Object Detection)

### 一阶段检测器 (One-stage)
- **YOLO (You Only Look Once)**: 将检测任务视为回归问题，极速推理。
- **来源**: [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)

### 二阶段检测器 (Two-stage)
- **Faster R-CNN**: 引入区域提议网络 (RPN)，精度更高。

## 3. 来源参考
- [Papers with Code - Image Classification](https://paperswithcode.com/task/image-classification)
