# ForensiQ — AI-Powered Image Forgery Detection Suite

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Framework-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**ForensiQ** is a high-performance image authenticity intelligence platform. It combines traditional forensic techniques like **Error Level Analysis (ELA)** with modern **Deep Learning (CNN)** architectures to detect splices, clones, and digital retouching with surgical precision.

---

## 🚀 Key Features

- 🔍 **Deep Neural Inference** — Utilizes a custom-trained 6-layer CNN architecture optimized for ELA feature extraction.
- 🌈 **ELA Heatmap Visualization** — Real-time generation of Error Level Analysis maps to visualize compression discrepancies.
- 📍 **Manipulation Mapping** — Automatically highlights specific regions of interest (ROI) where forgery is detected.
- 📊 **Confidence Scoring** — Provides a probabilistic score of authenticity (Real vs. Forged) for every analysis.
- 💎 **Premium UI/UX** — A sleek, modern dashboard designed for forensic experts and enthusiasts alike.

---

## 🧠 How It Works

### 1. Error Level Analysis (ELA)
Digital images lose a specific amount of data every time they are saved in a lossy format (like JPEG). When an image is modified, the modified part will have a different "error level" compared to the original. ForensiQ resaves the image at a known quality (90%) and calculates the absolute difference, scaling it to reveal hidden artifacts.

### 2. Neural Classification
The ELA-processed image is fed into a **Convolutional Neural Network (CNN)**:
- **Input Layer**: 128x128 RGB ELA-Map.
- **Backbone**: Multiple Conv2D layers with Batch Normalization and Dropout for robust feature learning.
- **Output**: Binary classification (0: Forged, 1: Authentic).

---

## 🛠️ Tech Stack

- **Backend**: Python 3.8+, Flask
- **Machine Learning**: TensorFlow 2.x, Keras
- **Image Processing**: Pillow (PIL), NumPy, OpenCV
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), Modern JavaScript

---

## 📦 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/forgery_detector.git
   cd forgery_detector
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Deploy Model Weights**
   Ensure `weights.weights.h5` is present in the root directory. This file contains the pre-trained neural weights for the 128x128 architecture.

4. **Launch Application**
   ```bash
   python app.py
   ```
   Navigate to `http://localhost:5000` in your browser.

---

## 📂 Project Structure

```text
forgery_detector/
├── app.py                  # Core Flask Application & API
├── weights.weights.h5      # Pre-trained CNN Weights (128x128)
├── requirements.txt        # System Dependencies
├── static/                 # CSS, JS, and UI Assets
└── templates/              # HTML Frontend (index.html)
```

---

## 🛡️ Disclaimer
This tool is intended for educational and research purposes. While highly effective at detecting common JPEG-based forgeries, forensic analysis should always involve multiple methods for conclusive results.

---

## 🤝 Contributing
Contributions are welcome! If you have ideas for improving the ELA algorithm or model architecture, feel free to open an issue or submit a pull request.

**Developed by Antigravity AI**
