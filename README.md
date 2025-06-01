# 🩺 Lung Cancer Detection using CT Scans (Deep Learning)

This project focuses on **early and accurate detection of lung cancer** using **CT scan images** and **deep learning models** such as **CNN**, **ResNet**, and an **Advanced CNN architecture**. Achieving up to **98% accuracy**, it aims to assist radiologists with faster and more reliable diagnostics.

---

## 📌 Features

- ✅ High-accuracy lung cancer classification (up to 98%)
- 🧠 Models: CNN, ResNet-50, Custom Advanced CNN
- 🗂️ CT scan image processing and augmentation
- 📈 Performance evaluation: Accuracy, Precision, Recall, F1-score
- 📊 Training visualization and confusion matrix
- 🌐 Gradio web interface for real-time predictions

---

## 🧠 Models Used

### 1. Basic CNN  
- 3 convolutional layers, ReLU, MaxPooling, Dropout  
- Achieved ~90% accuracy

### 2. ResNet-50  
- Transfer learning with ImageNet weights  
- Fine-tuned final layers  
- Achieved ~96% accuracy

### 3. Advanced CNN  
- Custom deep architecture with batch normalization  
- Data augmentation applied  
- Achieved **98% accuracy**

---

## 📁 Dataset

- **Dataset Used**: [IQ-OTH/NCCD – Iraq-Oncology Teaching Hospital/National Center for Cancer Diseases]([https://www.kaggle.com/datasets/andrewmvd/lung-cancer-dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset))  
- The dataset contains CT scan images categorized into:
  - **Normal**
  - **Benign**
  - **Malignant**

- **Preprocessing Steps**:
  - Image resizing (e.g., 224x224)
  - Normalization (0-1 scaling)
  - Data augmentation (rotation, zoom, horizontal flip)

- Data split into training, validation, and test sets using an 80/10/10 ratio.

---

## 📊 Performance Summary

| Model         | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| Basic CNN     | 90%      | 89%       | 88%    | 88.5%    |
| ResNet-50     | 96%      | 95%       | 94%    | 94.5%    |
| Advanced CNN  | **98%**  | **97.5%** | **97%**| **97.2%** |

---
## Images:
![image alt](https://github.com/Ranjana124/Lung-Cancer-Detection-And-Prediction-Using-CTscan/blob/main/images/Screenshot%202025-04-15%20120816.png?raw=true)
![image alt](https://github.com/Ranjana124/Lung-Cancer-Detection-And-Prediction-Using-CTscan/blob/main/images/Screenshot%202025-04-15%20120845.png?raw=true)
---
## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras, PyTorch  
- OpenCV, NumPy, Pandas  
- Matplotlib, Seaborn  
- Jupyter Notebook / Google Colab  
- **Gradio** for deployment

---

## 🚀 Gradio Deployment (Interactive Interface)

We provide a Gradio-based web interface for real-time predictions.
---

### 📦 Installation

```bash
pip install gradio

