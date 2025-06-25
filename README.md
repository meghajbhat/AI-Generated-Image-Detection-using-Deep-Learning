# 🧠 AI-Generated Image Detection using Deep Learning

> Detecting AI-generated synthetic images using a hybrid deep learning ensemble of ResNet and GAN-based discriminators.

## 📌 Overview

The rapid rise of AI-generated media—from tools like StyleGAN, DALL·E, and Stable Diffusion—has created urgent needs for verifying digital content authenticity. This project presents a robust deep learning framework capable of distinguishing AI-generated images from real ones using an ensemble of:

- 🟢 ResNet18 (CNN baseline)
- 🔵 ICGAN Discriminator
- 🔴 DCGAN Discriminator
- 🟣 StyleGAN Discriminator

Each model is fine-tuned and fused using a weighted softmax ensemble to maximize accuracy, generalization, and explainability.

---

## 🧪 Key Features

✅ **Hybrid Architecture**: Combines CNN and GAN-based discriminators  
✅ **Ensemble Learning**: Softmax-weighted score fusion across 4 models  
✅ **Cross-Domain Generalization**: Trained on CIFAKE and Kaggle datasets  
✅ **Grad-CAM Explainability**: Visualize model decisions  
✅ **Robust Against Perturbations**: Tested on noise, JPEG compression, and rotation  
✅ **94–98% Accuracy** per model, **97.26%** for the full ensemble  

---

## 🧰 Tech Stack

- Python 3.8+
- PyTorch
- torchvision
- Matplotlib / OpenCV
- Grad-CAM (for explainability)
- Git LFS (for large `.pth` model files)

---

## 🧬 Model Architecture

### 🟢 ResNet18
- Standard CNN used as a baseline classifier
- Fine-tuned on CIFAKE and tested on diverse datasets
- Pretrained weights + custom final layer

### 🔵 ICGAN, 🔴 DCGAN, 🟣 StyleGAN Discriminators
- Adapted and repurposed GAN discriminators as classifiers
- Fine-tuned to detect synthetic anomalies
- Regularization: label smoothing, noise injection, augmentation

### 🤝 Ensemble
- Weighted softmax probability fusion
- Final classifier achieves highest performance and robustness

---

## 🗂️ Datasets Used

### 📁 CIFAKE (Balanced AI vs Real Images)
- 120,000 images (60k real, 60k fake)
- Fake images generated using Stable Diffusion v1.4

### 📁 Kaggle AI vs Real
- 60,000 high-res images
- 30,000 AI-generated (MidJourney, DALL·E, SD)
- 30,000 real (Unsplash, WikiArt, Pexels)

---

## 📈 Performance Summary

| Model                        | Accuracy | F1 Score | ROC-AUC |
|-----------------------------|----------|----------|---------|
| ResNet18                    | 98.02%   | 0.98     | 0.9802  |
| ICGAN Discriminator         | 94.19%   | 0.94     | 0.9858  |
| DCGAN Discriminator         | 94.71%   | 0.95     | 0.9883  |
| StyleGAN Discriminator      | 96.89%   | 0.97     | 0.9959  |
| Ensemble (All 4 Models)     | **97.26%** | **0.97** | **0.9966** |

---

## 🧠 Explainability with Grad-CAM

Grad-CAM visualizations are used to identify which image regions contributed most to the model's classification—crucial for real-world forensics and transparency.

---

## 🛠️ How to Run

1. **Clone the repository**  
   ```bash
   git clone https://github.com/meghajbhat/AI-Generated-Image-Detection-using-Deep-Learning.git
   cd AI-Generated-Image-Detection-using-Deep-Learning
