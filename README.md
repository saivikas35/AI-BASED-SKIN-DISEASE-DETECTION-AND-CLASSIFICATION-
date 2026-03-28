# 🛡️ DermAI — AI-Based Skin Disease Detection & Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

**DermAI** is a state-of-the-art diagnostic assistant that utilizes a hybrid **ResNet-50 + Calibrated SVM** architecture to detect and classify **23 distinct skin disease categories**. By combining deep learning with clinical datasets (DermNet NZ & HAM10000), it provides dermatologists and patients with high-confidence predictions and visual "Explainable AI" heatmaps.

---

## 🚀 Key Features

*   **🧪 Hybrid AI Engine**: Uses **ResNet-50** for feature extraction and **Calibrated SVM** (23 classes) for maximum classification precision.
*   **🔍 Explainable AI (Grad-CAM)**: Generates heatmaps to show exactly which region of the skin lesion influenced the AI's diagnosis.
*   **📊 Confidence Scoring**: Provides a percentage-based reliability score for every prediction using probability calibration.
*   **📄 Medical Reporting**: Automatically generates detailed reports including descriptions, causes, and treatment suggestions.
*   **📱 Modern Dashboard**: A fully responsive, dark-themed UI for seamless image uploads and result viewing.

---

## 🛠️ Technology Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Deep Learning** | TensorFlow / Keras (ResNet-50) |
| **Machine Learning** | Scikit-learn (Calibrated SVM) |
| **Computer Vision** | OpenCV (CLAHE Pre-processing) |
| **Backend** | Flask |
| **Frontend** | HTML5, CSS3, Vanilla JS |

---

## 🔬 Methodology

### 1. Data Fusion
We integrated two massive dermatological repositories:
*   **DermNet NZ**: 23 categories of common and complex skin diseases.
*   **HAM10000 / ISIC**: High-risk pigmented lesions (Skin Cancer).

### 2. Pre-processing (CLAHE)
Images are normalized using **Contrast Limited Adaptive Histogram Equalization** to ensure consistent lighting and skin tone analysis across different datasets.

### 3. The Hybrid Algorithm
Instead of a standard Softmax layer, we extract a **2048-dimensional feature vector** from the final ResNet layer and feed it into a **Calibrated SVM** classifier. This provides superior decision boundaries for medically similar conditions.

---

## 📋 Supported Diseases (23 Classes)

*   Acne & Rosacea
*   Skin Cancer (Actinic Keratosis/BCC/Melanoma)
*   Atopic Dermatitis (Eczema)
*   Bacterial Infections (Cellulitis/Impetigo)
*   Fungal Infections (Ringworm/Nail Fungus)
*   Pigmentation Disorders
*   Viral Infections (Warts/Herpes)
*   Psoriasis & Lichen Planus
*   ...and 15 others.

---

## 💻 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/saivikas35/AI-BASED-SKIN-DISEASE-DETECTION-AND-CLASSIFICATION-.git
    cd AI-BASED-SKIN-DISEASE-DETECTION-AND-CLASSIFICATION-
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    python skin_disease_detection/app.py
    ```

4.  **Access the Dashboard**
    Open `http://127.0.0.1:5000` in your browser.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👤 Developed By
**Sai Vikas**  
*AI & Machine Learning Developer*

---

> **Note**: This tool is for educational and clinical support purposes only. Always consult a board-certified dermatologist for medical concerns.
