# 🎭 Multimodal Emotion Detection System (Audio + Facial Expressions)

This project implements a **multimodal emotion detection system** that predicts human emotions by combining **speech audio** and **facial expressions** using deep learning models.

The system integrates **audio-based emotion recognition** and **visual-based emotion recognition** into a single unified model for improved performance.

---

## 📌 Supported Emotions

The model classifies emotions into the following categories:

- Happy (HAP)
- Sad (SAD)
- Angry (ANG)
- Fear (FEA)
- Disgust (DIS)
- Neutral (NEU)

---

## 🧠 Model Architecture

This project uses a **three-part architecture**:

### 1️⃣ Voice Emotion Model
- **Model:** Wav2Vec2 (`facebook/wav2vec2-base`)
- Extracts speech representations from raw audio
- Applies temporal mean pooling and a classification head

### 2️⃣ Face Emotion Model
- **Model:** ResNet-50
- Processes facial images extracted from video frames
- Fine-tuned for emotion classification

### 3️⃣ Fusion Model
- Combines outputs from both audio and face models
- Uses feature concatenation followed by a fully connected layer
- Produces the final emotion prediction

---

## 📂 Dataset

- **CREMA-D Dataset Dataset sourced from Kaggle (https://www.kaggle.com/datasets/ejlok1/cremad)**
- Audio and video samples were aligned using filename-based matching
- Facial frames were extracted from video clips
- Data was split into training and testing sets

---


## 🧪 Model Training

Training was performed using **PyTorch** on **GPU-enabled Kaggle notebooks**.

### Training Pipeline
- Dataset loading and preprocessing
- Audio–face data pairing
- Separate training of:
  - Voice emotion model
  - Face emotion model
- Evaluation on validation datasets
- Multimodal fusion model training

All training logic is contained in:

training/model_development.ipynb


---

## 📊 Evaluation

- Audio-only and face-only models were evaluated independently
- The combined multimodal model was evaluated using paired audio–face samples
- Multimodal fusion demonstrated improved performance compared to unimodal approaches

---

## ▶️ Running Inference (Real-Time Emotion Detection)

### Requirements
- Webcam
- Microphone
- Python 3.8 or higher

### Steps
1. Place trained model weights inside the `models/` directory
2. Navigate to the project root directory
3. Run the inference script:

```bash
python inference/inference.py

# Inference Workflow

Captures a single image frame from the webcam

Records 4 seconds of audio from the microphone

Processes audio and image inputs using trained models

Produces a final emotion prediction in real time

# 🛠️ Technologies Used

Python

PyTorch

TorchVision

Torchaudio

Hugging Face Transformers

OpenCV

NumPy

PIL (Python Imaging Library)

SoundDevice

# 🚀 Skills and Concepts Demonstrated

Multimodal machine learning

Deep learning model fusion

Audio signal processing

Computer vision with convolutional neural networks (CNNs)

Transfer learning

Feature extraction using Wav2Vec2

Dataset preprocessing and alignment

Real-time inference system design

GPU-based model training

Model optimization and memory management

End-to-end machine learning pipeline development

Research-oriented project structuring



This project was developed as part of a professional portfolio to demonstrate practical expertise in:

Machine learning

Computer vision

Speech processing
