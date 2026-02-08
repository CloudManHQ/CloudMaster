# 生成模型 (Generative Models)

生成模型旨在学习数据的分布并生成全新的样本，是 AIGC 的核心技术。

## 1. 扩散模型 (Diffusion Models)

### 工作原理
- **前向过程 (Forward Process)**: 逐步在图像中加入高斯噪声。
- **反向过程 (Reverse Process)**: 训练神经网络（通常是 U-Net）从噪声中预测并去除噪声。
- **采样算法**: DDPM, DDIM。

### 应用
- **Stable Diffusion**: 基于潜在空间 (Latent Space) 的高效图像生成。
- **ControlNet**: 为扩散模型提供精细化控制（如姿态、边缘）。

## 2. GANs 与 VAEs
- **生成对抗网络 (GANs)**: 通过生成器与判别器的博弈进行训练。
- **变分自编码器 (VAEs)**: 通过变分推断学习隐变量分布。

## 3. 来源参考
- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Stable Diffusion Public Release](https://stability.ai/blog/stable-diffusion-public-release)
