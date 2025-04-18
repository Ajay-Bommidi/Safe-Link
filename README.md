# 🛡️ SafeLink - Hybrid ML-Based Phishing URL Detection System

SafeLink is a powerful Machine Learning-based system designed to detect phishing and malicious URLs in real-time. Using a hybrid model architecture with over 4 advanced ML algorithms and third-party threat intelligence APIs (Google Safe Browsing + VirusTotal), SafeLink aims to combat digital threats like phishing, financial fraud, and spam with high accuracy and reliability.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Hybrid%20Model-orange)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🚀 Table of Contents

- [🔍 Problem Statement](#-problem-statement)
- [🧠 Approach & Architecture](#-approach--architecture)
- [🧰 Tech Stack](#-tech-stack)
- [📊 Model Performance](#-model-performance)
- [📦 Installation](#-installation)
- [📁 Project Structure](#-project-structure)
- [📌 Future Work](#-future-work)
- [📝 License](#-license)

---

## 🔍 Problem Statement

Phishing and malicious URLs are responsible for a significant portion of digital crimes—including data breaches, financial fraud, and social engineering attacks. Traditional detection methods struggle with evolving threats. SafeLink is designed to address this challenge using advanced machine learning and real-time threat intelligence.

---

## 🧠 Approach & Architecture

SafeLink follows a multi-stage detection pipeline:

1. **Data Collection & Merging**
2. **Data Cleaning & Preprocessing**
3. **Feature Extraction**
4. **Data Balancing (SMOTE)**
5. **Explainability with SHAP**
6. **Hybrid Model Training**  
   - Models: `XGBoost`, `LightGBM`, `CatBoost`, `Random Forest`, `MLP`
   - Tuned using `Optuna`
7. **Threat Intelligence Integration**
   - Google Safe Browsing API
   - VirusTotal API
8. **Real-Time Scoring & Classification**

📌 **Data Split**: 80% training / 20% testing  
📌 **Model Output**: Risk score + Label (`Benign`, `Malicious`, `Suspicious`)

---

## 🧰 Tech Stack

- **Language**: Python
- **Libraries**:  
  `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `pandas`, `numpy`, `joblib`, `optuna`, `shap`
- **APIs**:  
  - Google Safe Browsing  
  - VirusTotal  

---

## 📊 Model Performance

| Metric        | Value    |
|---------------|----------|
| Accuracy      | 94.16%   |
| Macro Avg     | 94.19%   |
| Weighted Avg  | 94.19%   |
| Total Samples | 268,344  |

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/Ajay-Bommidi/Safe-Link.git
cd SafeLink
