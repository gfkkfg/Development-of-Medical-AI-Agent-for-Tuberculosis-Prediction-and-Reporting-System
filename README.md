# 🫁 Tuberculosis Detection AI Agent with Explainability and Treatment Assistance

This project is an AI-powered medical assistant that automates the process of detecting **tuberculosis (TB)** from **chest X-ray images** using deep learning. The system not only classifies the presence of TB but also provides **interpretable visual explanations**, **treatment plan generation**, **PDF report creation**, and **email delivery to radiologists and patients** — making it a powerful tool for use in clinical and rural healthcare settings.

---

## 📌 Key Features

### 🔍 1. **Automated Tuberculosis Detection**
- Uses a **fine-tuned InceptionV3 Convolutional Neural Network (CNN)**.
- Classifies X-ray images into **TB positive** or **TB negative** categories.
- Handles image preprocessing such as resizing, normalization, and augmentation.

### 🧠 2. **Model Explainability with Grad-CAM**
- Integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)**.
- Generates **heatmaps overlayed** on X-ray images to visually highlight lung regions that influenced the model’s decision.
- Enhances **trust and interpretability** for radiologists and medical professionals.

### 💬 3. **AI-Powered Treatment Summary**
- Uses the **Groq Cloud API** (or LLM integration) to automatically generate **custom treatment recommendations** for TB-positive cases.
- Tailors output based on classification result and model confidence.

### 🧾 4. **PDF Report Generation**
- Automatically compiles:
  - Patient name and details
  - Classification result
  - Model confidence
  - Grad-CAM heatmap
  - Treatment plan
- Saves a high-quality, shareable PDF for recordkeeping.

### 📧 5. **Email Automation**
- Sends the generated report to:
  - The **radiologist** (for professional review)
  - The **patient** (for personal awareness)
- Uses **Yagmail** for secure and seamless Gmail-based delivery.

### 📋 6. **Feedback System for Model Retraining**
- Patients or radiologists can provide **feedback on model predictions**.
- Feedback is saved in a **CSV file** for future **model retraining or performance tuning**.

---

## 🛠️ Tech Stack

| Layer              | Tools/Frameworks Used                    |
|-------------------|-------------------------------------------|
| **Frontend (UI)** | Streamlit (interactive web interface)     |
| **Model**         | TensorFlow, Keras (InceptionV3 architecture) |
| **Explainability**| OpenCV, Grad-CAM                          |
| **Email**         | Yagmail, Gmail SMTP                       |
| **LLM Integration**| Groq Cloud API (or ChatGPT API)           |
| **Report Export** | ReportLab / FPDF for PDF generation       |
| **Data Storage**  | CSV, Pandas                               |

---

## 🧪 How to Run the Project Locally

### 🔧 Prerequisites
- Python ≥ 3.8
- A virtual environment (recommended)
- Packages listed in `requirements.txt`

### ⚙️ Setup Instructions

# Clone the repository
git clone https://github.com/gfkkfg/Development-of-Medical-AI-Agent-for-Tuberculosis-Prediction-and-Reporting-System.git
cd tb-ai-agent

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

Author: Thangaraj M
Linkedin: https://www.linkedin.com/in/thangarajsankar
Portfolio: https://thangaraj.onrender.com
