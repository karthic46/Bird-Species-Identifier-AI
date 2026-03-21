# Bird-Species-Identifier-AI

* AI-powered Bird Species Identification system using TensorFlow and Librosa to classify bird sounds based on MFCC audio features, with an interactive Streamlit dashboard.

* The model analyzes uploaded or recorded audio and predicts the bird species based on MFCC audio features.

Built using **TensorFlow, Librosa, and Streamlit**, and deployed as an interactive web application.

---

## 🚀 Live Demo
https://bird-species-identifier-ai-qi82svyxaieeiyo2jzr4wd.streamlit.app/

---

## 📌 Features

- 🎧 Upload bird audio files (WAV / MP3)
- 🎙 Record bird sound directly from microphone
- 🧠 Deep learning model prediction
- 📊 Model performance metrics visualization
- 🌐 Interactive web interface using Streamlit
- ⚡ Real-time audio feature extraction using MFCC

---

## 🧠 How It Works

1. User uploads or records bird audio
2. Audio is processed using **Librosa**
3. **MFCC features** are extracted
4. Features are passed to a **CNN deep learning model**
5. The model predicts the **bird species**
6. Prediction and confidence are displayed in the dashboard

## 🏗️ Project Architecture

Audio Input
│
▼
Feature Extraction (MFCC - Librosa)
│
▼
Deep Learning Model (TensorFlow CNN)
│
▼
Prediction
│
▼
Streamlit Web Interface

## 📂 Project Structure


Bird-Species-Identifier-AI
│
├── app.py # Streamlit application
├── requirements.txt # Python dependencies
├── runtime.txt # Python runtime version
├── README.md
│
├── model_files
│ ├── model.h5 # Trained CNN model
│ └── prediction.json # Bird labels
│
├── sample_audio # Example audio files
│
└── Inference_Images # Bird images for predictions


## 🛠️ Technologies Used

| Technology | Purpose                   |
|------------|---------------------------|
| Python     | Core programming language |
| TensorFlow | Deep learning model       |
| Librosa    | Audio processing          |
| NumPy      | Numerical operations      |
| Streamlit  | Web application           |
| Plotly     | Data visualization        |


## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/karthic46/bird-species-identifier-ai.git
cd bird-species-identifier-ai
```

Create and activate a local virtual environment (recommended on Windows):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Dependency policy:

- `requirements.txt` is pinned to tested versions for reproducible installs.
- If you want newer packages, upgrade intentionally and retest the app before deployment.

Quick health check (recommended):

```bash
python -c "import tensorflow as tf, librosa, streamlit as st; print('OK | tensorflow', tf.__version__, '| librosa', librosa.__version__, '| streamlit', st.__version__)"
```

Run the application:

```bash
python -m streamlit run app.py
```

### Windows note (TensorFlow install error)

If you see an error like `Could not install packages due to an OSError` with a very long path, you are likely hitting Windows path length limits (common with Microsoft Store Python paths).

Use these fixes:

1. Prefer the local `.venv` setup above instead of global/site-packages installs.
2. Enable long paths in Windows (admin PowerShell):

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

3. Restart Windows, then run install again in the project virtual environment.

**📊 Model Details**

Model Type: Convolutional Neural Network (CNN)
Feature Extraction: MFCC (Mel-Frequency Cepstral Coefficients)
Framework: TensorFlow / Keras
Input: Bird audio recordings
Output: Predicted bird species

**🔮 Future Improvements**

Add real-time bird sound detection
Improve model accuracy with larger dataset
Display bird image after prediction
Add top-3 prediction results
Mobile optimized interface

**👨‍💻 Author**

Karthick Raja

GitHub:
https://github.com/karthic46
