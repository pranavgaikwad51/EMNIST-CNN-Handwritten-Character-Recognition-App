# 🧠 EMNIST CNN Handwritten Character Recognition App

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![Model](https://img.shields.io/badge/Model-CNN-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-EMNIST-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

---

## 📌 Overview

This project implements a **handwritten character recognition system** using a **Convolutional Neural Network (CNN)** trained on the **EMNIST dataset**.  
The model supports recognition of:

- 0–9 digits  
- A–Z uppercase letters  
- a–z lowercase characters  

Users can draw directly on the canvas or upload an image to get predictions.  
The app displays **confidence score, top-5 predictions, and pre-processing steps** used for inference.

👉 **Live Application:**  
🔗 https://emnist-cnn-handwritten-character-recognition-app-pranavgaikwad.streamlit.app/

---

## 🧠 Problem Statement

Handwriting varies significantly across individuals. Building a system capable of understanding different writing styles, noise, thickness, and orientation is challenging.

This project solves that problem using a deep-learning model that recognizes handwritten characters with high accuracy.

---

## 🎯 Objective

- Develop a **robust CNN model** to classify handwritten characters across 62 categories.
- Build an **interactive UI** using Streamlit for real-time prediction.
- Provide **explainability features** such as preprocessing visualization and top probability predictions.

---

## 📁 Dataset

| Parameter | Details |
|----------|---------|
| Dataset | EMNIST (Extended MNIST) |
| Classes | 62 |
| Type | Grayscale images |
| Image Size | 28 × 28 |
| Source | National Institute of Standards and Technology (NIST) |

Dataset link:  
🔗 https://www.nist.gov/itl/products-and-services/emnist-dataset

---

## 🛠 Tools & Libraries

| Category | Technology |
|----------|-----------|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Deployment | Streamlit |
| Data Processing | NumPy, Pandas |
| Image Processing | Pillow (PIL), streamlit-drawable-canvas |

---

## 🧬 Model Architecture

Input (28x28 grayscale image)
↓
Conv2D + ReLU Activation
↓
MaxPooling
↓
Conv2D + ReLU
↓
MaxPooling
↓
Flatten Layer
↓
Dense Layer (ReLU)
↓
Dropout (to avoid overfitting)
↓
Output Layer (Softmax - 62 classes)


---

## 🔧 Data Preprocessing Pipeline

✔ Noise removal  
✔ Bounding box cropping  
✔ Image resizing → 28×28  
✔ Contrast enhancement  
✔ Normalization → values scaled to [0, 1]  
✔ Final reshaping → `(1, 28, 28, 1)` before prediction  

---

## 📈 Evaluation Metrics

| Metric | Value |
|--------|-------|
| Training Accuracy | ~95–97% |
| Testing Accuracy | ~92–96% |
| Loss | ~0.28 |

Top-5 prediction confidence visualization included.

---

## 🖥 User Interface Screenshots

> *(You can add images later to show your UI example)*
📎 /screenshots
├── interface.png
├── prediction_output.png
└── preprocessing_steps.png


---

## 🚀 Running the Project Locally

### 1️⃣ Clone the Repository
```bash
git clone <your-repository-link>
cd EMNIST-CNN-Handwritten-Character-Recognition-App

2️⃣ Install Required Libraries
pip install -r requirements.txt

3️⃣ Run the Application
streamlit run app.py

🧾 Files Included
File	                                   Description
app.py	                                Streamlit UI + Model Execution Code
cnn_emnist_digits_alphabets.pkl	        Trained ML Model
README.md	                              Project Documentation
requirements.txt	                      Dependencies File

❤️ Acknowledgements

EMNIST Dataset

Streamlit Community

TensorFlow Framework

📜 License

This project is licensed under the MIT License.
Feel free to use and modify for learning and research purposes.

👤 Developer Information
Field	                      Details
Name	                      Pranav Gaikwad
Email	                      📧 gaikwadpranav988@gmail.com

LinkedIn	                  🔗 https://www.linkedin.com/in/pranav-gaikwad-0b94032a

GitHub	                    🧠 https://github.com/pranavgaikwad51

Streamlit App	               🚀 https://emnist-cnn-handwritten-character-recognition-app-pranavgaikwad.streamlit.app/

Phone	                        📱 7028719844
