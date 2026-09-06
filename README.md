# Parkinson's Disease Detection from Speech

A Flask + Scikit-learn educational project for Parkinson's disease classification using 22 extracted voice features.

## Current version

**Important:** this version does not directly analyze a `.wav` recording. The model expects the 22 numerical voice features used by the training dataset.

Pipeline:

`22 voice features -> Min-Max scaling -> KNN -> prediction`

## Project structure

```text
Parkinsons_disease_detection_from_speech_project/
├── app.py
├── train_model.py
├── data.csv
├── requirements.txt
├── models/
│   ├── parkinsons_knn_pipeline.pkl
│   └── metadata.json
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── script.js
```

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

Open:

```text
http://127.0.0.1:5000
```

Do **not** open `index.html` directly and do not use Live Server.

## Model information

Dataset rows used: 195

The training script:
- separates the 22 input features from `status`
- uses a stratified 75/25 evaluation split
- scales only the input features
- uses 10-fold stratified cross-validation
- evaluates KNN with accuracy and MCC
- trains the final KNN pipeline on the complete cleaned dataset
- saves the complete preprocessing + model pipeline

## Medical disclaimer

This is an educational/research project. It is not a medical diagnostic device and should not replace professional medical advice.

## Future upgrade: real speech upload

To make the project genuinely accept a microphone recording or `.wav` file, add an audio feature-extraction layer (for example with Librosa) and train on a compatible audio dataset. The extracted feature set must match the model's training features.
