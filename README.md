# 🧠 Parkinson's Disease Detection from Speech

A machine learning-based web application that classifies Parkinson's-related patterns using **22 extracted biomedical voice features**.

The project uses a **K-Nearest Neighbors (KNN)** classification algorithm with **MinMaxScaler** for feature scaling. A Flask-based web application provides an interactive interface for entering voice features and viewing prediction results.

> ⚠️ **Disclaimer:** This project is developed for educational and research purposes only. It is not a medical diagnostic system and should not replace professional medical advice.

---

## 📌 Project Overview

Parkinson's disease can affect speech and voice characteristics. Features such as frequency variation, jitter, shimmer, HNR, and other acoustic measurements can be used for machine learning research.

This project uses **195 voice samples** with **22 voice-related features** to classify samples into two classes:

- `0` → Healthy / Control
- `1` → Parkinson's-related class

The trained model is integrated with a Flask web application.

---

## 🎯 Objectives

- Analyze biomedical voice features.
- Build a machine learning classification model.
- Use KNN for Parkinson's-related pattern classification.
- Apply Min-Max feature scaling.
- Develop an interactive web interface.
- Display prediction and model probability.
- Integrate the ML model with Flask.

---

## 🛠️ Technologies Used

- **Python**
- **Flask**
- **Scikit-learn**
- **K-Nearest Neighbors (KNN)**
- **Pandas**
- **NumPy**
- **Joblib**
- **HTML5**
- **CSS3**
- **JavaScript**
- **Git & GitHub**
- **Visual Studio Code**

---

## 📊 Dataset

The dataset contains:

- **195 samples**
- **22 biomedical voice features**
- **1 target variable (`status`)**

The 22 features include frequency, jitter, shimmer, HNR, RPDE, DFA, spread1, spread2, D2, PPE, and other voice measurements.

---

## 🤖 Machine Learning Pipeline

```text
22 Voice Features
       ↓
Min-Max Scaling
       ↓
KNN Classifier
       ↓
Prediction
       ↓
Model Probability

---

## 📂 Project Structure
```text
Parkinsons-Disease-Detection-from-Speech/
│
├── models/
│   ├── parkinsons_knn_pipeline.pkl
│   └── metadata.json
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
├── templates/
│   └── index.html
│
├── app.py
├── train_model.py
├── data.csv
├── requirements.txt
├── README.md
└── .gitignore
```
---

## Run in VS Code

Open the project folder in VS Code and run:

```bash
python -m venv venv
```

Windows PowerShell:

```bash
venv\Scripts\Activate.ps1
```

Install packages:

```bash
pip install -r requirements.txt
```

The trained model is already included. If you change `data.csv`, retrain it:

```bash
python train_model.py
```

Start the website:

```bash
python app.py
```

---

## Future upgrade: real speech upload

To make the project genuinely accept a microphone recording or `.wav` file, add an audio feature-extraction layer (for example with Librosa) and train on a compatible audio dataset. The extracted feature set must match the model's training features.
