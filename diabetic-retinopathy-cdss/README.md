# 🩺 Diabetic Retinopathy Clinical Decision Support System (DR-CDSS)

An AI-powered Clinical Decision Support System for automated diabetic retinopathy screening from retinal fundus images. The system combines deep learning–based image classification with Retrieval-Augmented Generation (RAG) to provide explainable, evidence-backed clinical recommendations.

> **⚠️ Disclaimer:** This tool is designed for **screening assistance only** and does **not** replace professional clinical diagnosis. All predictions must be reviewed and confirmed by a qualified ophthalmologist. The system is intended to support — not substitute — clinical decision-making.

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [License](#license)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                       │
│         (Image Upload · Results Dashboard · PDF Reports)        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                     ┌─────────▼─────────┐
                     │   FastAPI Backend  │
                     │  (REST API Layer)  │
                     └────┬─────────┬────┘
                          │         │
              ┌───────────▼──┐  ┌───▼───────────┐
              │  DL Pipeline │  │  RAG Pipeline  │
              │              │  │                │
              │ EfficientNet │  │ Knowledge Base │
              │   B3 Model   │  │  (PDF/Text)    │
              │              │  │                │
              │  Grad-CAM    │  │ FAISS Vector   │
              │ Explainability│ │   Database     │
              └──────────────┘  │                │
                                │ OpenAI LLM     │
                                │ (GPT-4 / 3.5)  │
                                └────────────────┘
```

**Key Components:**

1. **Deep Learning Pipeline** — EfficientNet-B3 fine-tuned on retinal fundus images for 5-class DR severity grading (No DR → Proliferative DR), with Grad-CAM heatmaps for visual explainability.
2. **RAG Pipeline** — Retrieval-Augmented Generation using clinical literature embeddings stored in FAISS, queried by an LLM to produce evidence-based recommendations.
3. **FastAPI Backend** — RESTful API orchestrating image analysis, RAG queries, and PDF report generation.
4. **Streamlit Frontend** — Interactive dashboard for image upload, real-time predictions, Grad-CAM visualizations, and downloadable clinical reports.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Deep Learning** | TensorFlow / Keras, EfficientNet-B3, OpenCV |
| **RAG / NLP** | Sentence-Transformers, FAISS, OpenAI GPT |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Streamlit, Plotly |
| **Reporting** | ReportLab (PDF generation) |
| **Data** | Pandas, NumPy, scikit-learn, Matplotlib, Seaborn |

---

## 📁 Project Structure

```
diabetic-retinopathy-cdss/
├── dataset/                    # Retinal fundus image dataset (not tracked)
│   ├── train_images/
│   ├── test_images/
│   └── train.csv
├── knowledge_base/             # Clinical PDFs & reference texts for RAG
├── models/                     # Saved model weights (.h5)
├── vector_db/                  # FAISS vector store (auto-generated)
├── reports/                    # Generated PDF clinical reports
├── static/
│   ├── uploads/                # Uploaded fundus images
│   └── gradcam/                # Grad-CAM heatmap outputs
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py           # Configuration & environment management
│   ├── data_preprocessing.py   # Image loading, augmentation, pipeline
│   ├── model.py                # EfficientNet-B3 model definition & training
│   ├── gradcam.py              # Grad-CAM heatmap generation
│   ├── rag_engine.py           # RAG: embeddings, FAISS, LLM queries
│   ├── report_generator.py     # PDF report generation (ReportLab)
│   └── api.py                  # FastAPI application & endpoints
├── app.py                      # Streamlit frontend application
├── train.py                    # Model training entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)
- **Git**
- **Kaggle API credentials** (for dataset download)

### Steps (Windows PowerShell)

```powershell
# 1. Clone or navigate to the project
cd "c:\Users\tanma\OneDrive\Desktop\daibatic retinapathy detection\diabetic-retinopathy-cdss"

# 2. Create a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
Copy-Item .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Download the dataset (see Dataset section below)
```

---

## 📊 Dataset

This project uses the [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) dataset from Kaggle.

### Download Instructions

1. **Set up Kaggle API credentials:**
   - Go to [kaggle.com/settings](https://www.kaggle.com/settings) → Create New Token
   - Place the downloaded `kaggle.json` in `C:\Users\<YourUser>\.kaggle\`

2. **Download the dataset:**
   ```powershell
   kaggle competitions download -c aptos2019-blindness-detection -p dataset/
   ```

3. **Extract:**
   ```powershell
   Expand-Archive -Path dataset\aptos2019-blindness-detection.zip -DestinationPath dataset\
   ```

### DR Severity Classes

| Grade | Label | Description |
|-------|-------|-------------|
| 0 | No DR | Healthy retina, no abnormalities |
| 1 | Mild NPDR | Few microaneurysms |
| 2 | Moderate NPDR | Hemorrhages, hard exudates, vessel blockage |
| 3 | Severe NPDR | Widespread hemorrhages, venous beading, IRMA |
| 4 | Proliferative DR | Neovascularization, high risk of vision loss |

---

## ▶️ Usage

### 1. Train the Model

```powershell
python train.py
```

This will preprocess images, train the EfficientNet-B3 model, and save weights to `models/`.

### 2. Build the Knowledge Base

```powershell
python -c "from src.rag_engine import build_vector_store; build_vector_store()"
```

Place clinical PDFs and reference documents in `knowledge_base/` before running.

### 3. Start the API Server

```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

API documentation is available at `http://localhost:8000/docs`.

### 4. Launch the Streamlit Dashboard

```powershell
streamlit run app.py
```

Open `http://localhost:8501` in your browser to upload fundus images and view predictions.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) competition on Kaggle
- EfficientNet architecture by Tan & Le (2019)
- Clinical grading standards from the International Clinical Diabetic Retinopathy (ICDR) scale

---

> **Remember:** AI-assisted screening is a tool to aid clinicians — it should never be the sole basis for diagnosis or treatment decisions. Always consult a qualified healthcare professional.
