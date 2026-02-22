# 🧠 Handwritten Digit Recognition using CNN (MNIST) – 99.36% Accuracy

This project implements and deploys a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) using the MNIST dataset. The final model achieves **99.36% test accuracy** and is deployed using **Streamlit** for real-time digit prediction through an interactive drawing interface.

---
## 🎥 Live Demo

<p align="center">
  <img src="assets/demo.gif" width="650">
</p>

---

## 🚀 Project Overview

The objective of this project was to:

- Understand CNN fundamentals for image classification
- Build a clean deep learning pipeline
- Improve model performance using architectural enhancements
- Evaluate the model using professional metrics
- Deploy the trained model as a real-time web application

This project demonstrates the complete workflow from model training to deployment.

---

## 📦 Dataset

**Dataset Used:** MNIST  
- 60,000 training images  
- 10,000 test images  
- Image size: 28×28 pixels  
- Grayscale images  
- 10 output classes (digits 0–9)

Each image contains a centered handwritten digit.

---

## 🏗️ Model Architecture

The final CNN architecture includes:

- Conv2D (32 filters, 3x3, ReLU)
- BatchNormalization
- Conv2D (32 filters, 3x3, ReLU)
- BatchNormalization
- MaxPooling2D
- Dropout (0.25)

- Conv2D (64 filters, 3x3, ReLU)
- BatchNormalization
- Conv2D (64 filters, 3x3, ReLU)
- BatchNormalization
- MaxPooling2D
- Dropout (0.25)

- Flatten
- Dense (128 units, ReLU)
- BatchNormalization
- Dropout (0.5)
- Dense (10 units, Softmax)

### Why These Improvements?

- **BatchNormalization** stabilizes and accelerates training.
- **Dropout** prevents overfitting.
- **Deeper convolution blocks** improve feature extraction.
- **EarlyStopping** avoids unnecessary training once validation performance plateaus.

---

## 📊 Model Performance

- Training Accuracy: ~99%
- Validation Accuracy: ~99%
- Test Accuracy: **99.36%**
- Precision, Recall, and F1-score ≈ 0.99 across all classes
- Strong diagonal confusion matrix (minimal misclassification)

The model demonstrates excellent generalization performance.

---

## 🧠 Key Learnings

### 1️⃣ CNN Fundamentals
- How convolution extracts spatial features
- Importance of pooling for dimensionality reduction
- Role of softmax in multi-class classification

### 2️⃣ Importance of Preprocessing
- Normalization (0–255 → 0–1) is critical
- Channel dimension required for Conv2D
- Deployment preprocessing must match training preprocessing

### 3️⃣ Debugging Distribution Shift
During deployment, prediction inconsistencies occurred due to differences between MNIST-style digits and custom-drawn digits. This highlighted the importance of:
- Input consistency
- Stroke thickness alignment
- Centering and scaling
- Matching deployment data distribution with training data

### 4️⃣ Real-World ML Insight
High test accuracy does not guarantee perfect real-world performance if deployment input distribution differs from training data.

---

## 🌐 Deployment (Streamlit App)

The trained model is deployed using **Streamlit** with an interactive drawing canvas.

### Features:
- Draw digit directly in browser
- Automatic preprocessing
- Real-time prediction
- Confidence score display
- Probability distribution bar chart
- Clean professional UI with sidebar summary
