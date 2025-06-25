# AI-Generated Image Detection using Deep Learning

## 📄 Project Summary

This repository contains a comprehensive deep learning framework to detect AI-generated synthetic images using a combination of convolutional neural networks and GAN-based discriminator models. The project is developed as part of a final-year capstone at PES University, Bangalore.

The goal is to accurately classify images as real or AI-generated, using advanced deep learning architectures trained and evaluated on two major datasets. It also integrates explainability via Grad-CAM and demonstrates ensemble learning for enhanced generalization across unseen generative models.

---

## 🔧 Features

* ✨ **Four deep learning models**: ResNet18, ICGAN, DCGAN, StyleGAN Discriminators
* ⚖️ **Softmax-weighted ensemble**: Aggregates predictions to enhance robustness
* 🔹 **Fine-tuned discriminators**: Pretrained on CIFAKE, tuned on Kaggle real vs AI image dataset
* 📊 **Grad-CAM visualizations**: Explainable AI using heatmaps
* ⚡ **Robustness tested**: Under image corruption (noise, JPEG compression, rotation)

---

## 🎓 Motivation

With the rise of generative models like DALL·E, MidJourney, and StyleGAN, distinguishing real images from AI-generated ones has become a critical need in journalism, forensics, and content moderation.

Traditional forensic techniques fall short against modern GANs and diffusion models. This project addresses this by combining both supervised CNNs and GAN-trained discriminators for better detection.

---

## 🔸 Architecture Overview

### 🗾 ResNet18 (Baseline CNN Classifier)

* Pretrained on ImageNet
* Fine-tuned on CIFAKE dataset
* Captures general visual features

### 🔷 GAN-Based Discriminators:

* **ICGAN**: Captures semantic inconsistencies
* **DCGAN**: Detects low-level artifacts like noise, pixel anomalies
* **StyleGAN**: Focuses on style, lighting, and high-level coherence

### 🤝 Ensemble Classifier

* Combines outputs using weighted probability fusion
* Weights optimized empirically
* Ensemble achieves best generalization and accuracy

---

## 📚 Datasets Used

### 1. CIFAKE

* 120,000 images (60k real + 60k fake)
* Fake images generated using Stable Diffusion v1.4
* Resolution: 32x32 (CIFAR-10 format)
* Purpose: Controlled training

### 2. Kaggle AI vs Real

* 60,000 images
  * 30k AI-generated (DALL·E, MidJourney, SD)
  * 30k real (Unsplash, Pexels, WikiArt)
* Diverse styles and high resolution
* Purpose: Real-world fine-tuning and validation

---

## 💪 Training Pipeline

### Common Preprocessing

* Resize to 32x32
* Normalization
* Augmentations: Horizontal flips, random crop

### Loss Function

* Binary Cross Entropy Loss

### Regularization

* Label Smoothing (real = 0.9)
* Gaussian Noise Injection
* Dropout Layers

### Optimization

* Adam Optimizer
* Learning Rate: 1e-4 – 3e-4
* Batch Size: 64
* Early Stopping

---

## 📊 Model Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

---

## 📉 Performance Summary

| Model                   | Accuracy   | F1 Score | ROC-AUC    |
|-------------------------|------------|----------|------------|
| ResNet18                | 98.02%     | 0.98     | 0.9802     |
| ICGAN Discriminator     | 94.19%     | 0.94     | 0.9858     |
| DCGAN Discriminator     | 94.71%     | 0.95     | 0.9883     |
| StyleGAN Discriminator  | 96.89%     | 0.97     | 0.9959     |
| Ensemble (All 4 Models) | **97.26%** | **0.97** | **0.9966** |

---

## 🔍 Grad-CAM Explainability

* Applied to ResNet18 and GAN discriminators
* Visualizes important regions influencing classification
* Useful in forensic analysis, courtroom applications

---

## 📎 Folder Structure

```
.
├── Code files/
│   ├── GAN training.ipynb
│   ├── ResNet training.ipynb
│   └── Ensemble logic.ipynb
├── finetuned files/
│   ├── *.pth (Model weights)
├── output screenshots/
│   ├── ResNet, DCGAN, ICGAN, StyleGAN results
├── Research Paper.pdf
├── README.md
```

---

## 📥 Setup Instructions

### Prerequisites

* Python 3.8+
* Git LFS
* PyTorch, torchvision

### Installation

```bash
# Clone repo
git clone https://github.com/meghajbhat/AI-Generated-Image-Detection-using-Deep-Learning.git
cd AI-Generated-Image-Detection-using-Deep-Learning

# Install dependencies
pip install -r requirements.txt

# Pull LFS files (model weights)
git lfs install
git lfs pull
```

---

## ⚒️ Running the Project

1. **Train individual models** (if not using pre-trained)

   ```bash
   python train_resnet.py
   python train_discriminators.py
   ```

2. **Run ensemble evaluation**

   ```bash
   python ensemble_predict.py
   ```

3. **Visualize results with Grad-CAM**

   ```bash
   python gradcam_visualize.py
   ```

---

## 🚀 Results Highlights

* Ensemble Accuracy: **97.26%**
* Strong performance under noise and JPEG compression
* ResNet18 alone achieves 98% on CIFAKE
* Grad-CAM confirms interpretable decision-making

---

## 🔬 Future Work

* Include SDXL, MidJourney V6 in training
* Introduce vision transformers into ensemble
* Create REST API or browser extension for real-time detection
* Explore contrastive learning & adversarial training

---

## 📜 Citation

```bibtex
@article{patra2024aigen,
  title={AI-Generated Image Detection using Deep Learning},
  author={Patra, Hrishita and Bobba, Jahnavi and K, Keerthi and Bhat, Megha and Narayan, Surabhi},
  journal={Capstone Project, PES University},
  year={2024}
}
```

---

## 🧳 Acknowledgments

* PES University, Bangalore
* Dr. Surabhi Narayan (Faculty Mentor)
* Kaggle Datasets & CIFAKE Authors
* PyTorch, GitHub, HuggingFace Community

---

## 🚑 Contact

* Megha Bhat: [meghajbhat@gmail.com](mailto:meghajbhat@gmail.com)
* Hrishita Patra: [hrishitapatra@gmail.com](mailto:hrishitapatra@gmail.com)
* Jahnavi Bobba: [janubobba1@gmail.com](mailto:janubobba1@gmail.com)
* Keerthi K: [keerthi2004kk@gmail.com](mailto:keerthi2004kk@gmail.com)

---

## ⭐ Contribute

Found a bug or have suggestions? Open an issue or pull request. Help improve AI-based media verification!

---

## 💫 License

MIT License. See [LICENSE](./LICENSE) for details.
