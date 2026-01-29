# 🎭 Real-Time Facial Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![YOLO](https://img.shields.io/badge/YOLO-Face%20Detection-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview
This project implements a **Real-Time Facial Emotion Recognition System** using Deep Learning and Computer Vision.  
The system detects human faces using **YOLO-based face detection** and classifies emotions using a **Convolutional Neural Network (CNN)** model.

The system recognizes emotions such as:
- 😊 Happy  
- 😡 Angry  
- 😢 Sad  
- 😲 Surprise  
- 😐 Neutral  

This project can be used in **Human-Computer Interaction, Mental Health Monitoring, Security Surveillance, and Smart AI Applications**.

---

## 🧠 Tech
- **Programming Language:** Python  
- **Libraries:** OpenCV, NumPy  
- **Deep Learning:** TensorFlow / Keras  
- **Face Detection:** YOLO / Haar Cascade  
- **Model:** CNN for Emotion Classification  

---

## 🚀 Features
- Real-time face detection from webcam  
- Emotion classification using CNN  
- Bounding box visualization with emotion labels  
- Live video processing using OpenCV  
- Modular and easy-to-understand code structure  

---

## 🏗️ System Architecture
1. Capture live video from webcam  
2. Detect faces using YOLO  
3. Crop detected face regions  
4. Classify emotions using CNN  
5. Display emotion labels on video feed
  
---

📊 Applications

Human-computer interaction systems
Mental health monitoring tools
Smart surveillance systems
Educational engagement analysis
Emotion-aware AI assistants
---

## ⚙️ How to Run
```bash
pip install -r requirements.txt
python main.py

