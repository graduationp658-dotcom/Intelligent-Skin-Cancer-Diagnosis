# 🔬 Intelligent Skin Cancer Diagnosis

> AI-powered dermoscopy image classifier with explainable AI and clinical decision support — graduation project.

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/graduationProject1/Skin-Cancer)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.17-orange)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](LICENSE)

---

## 🩺 Overview

This system classifies dermoscopy images into **8 skin lesion types**, explains its predictions with **Grad-CAM++ heatmaps**, and provides evidence-based clinical guidance through a **Retrieval-Augmented Generation (RAG)** medical assistant — all running fully locally with no external API calls.

| Component | Technology |
|-----------|------------|
| Classifier | EfficientNet-B4 + Squeeze-and-Excitation block |
| Explainability | Grad-CAM++ (class-specific heatmaps) |
| Knowledge Retrieval | FAISS + sentence-transformers (all-MiniLM-L6-v2) |
| Clinical LLM | Qwen2.5-0.5B-Instruct (GGUF, local CPU) |
| Interface | Gradio 6 (DermaScan AI) |

---

## 🎯 Key Features

- **8-class classification** — actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, squamous cell carcinoma, vascular lesion
- **Malignancy screening** — sensitivity-optimized Youden thresholds for melanoma (87.1%), BCC (94.5%), SCC (81.4%)
- **Grad-CAM++ visual explanation** — highlights the exact image region driving the prediction
- **Diagnostic report cards** — urgency level (URGENT / MONITOR / ROUTINE), clinical features, recommendation, differential diagnoses
- **Conversational AI assistant** — answers clinical questions grounded only in the local knowledge base (no hallucination, no internet)
- **Fully offline** — no cloud APIs, no telemetry, patient images never leave the machine

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Macro AUC | **0.873** |
| Melanoma Sensitivity | 87.1% (threshold-tuned) |
| BCC Sensitivity | 94.5% |
| SCC Sensitivity | 81.4% |
| Balanced Accuracy | ~72% |

Training followed a **3-stage progressive fine-tuning** strategy:
1. Head-only (backbone frozen)
2. Last 4 backbone blocks unfrozen
3. Full model fine-tuning at low LR

---

## 🗂️ Project Structure

```
graduation project/
│
├── 📓 Notebooks (Phases 1–8)
│   ├── skin_cancer_phase1.ipynb        # EDA & dataset analysis
│   ├── skin_cancer_phase2.ipynb        # Preprocessing pipeline (v2)
│   ├── skin_cancer_phase3.ipynb        # Baseline model
│   ├── skin_cancer_phase4.ipynb        # EfficientNet-B4 + SE block
│   ├── skin_cancer_phase4i.ipynb       # Best training run (AUC 0.873)
│   ├── skin_cancer_phase4_improved.ipynb # v6 dataset / improved augmentation
│   ├── skin_cancer_phase5.ipynb        # Evaluation & threshold tuning
│   ├── skin_cancer_phase6.ipynb        # Ensemble experiments
│   ├── skin_cancer_phase7_clinical_rag.ipynb # RAG pipeline construction
│   └── skin_cancer_phase8.ipynb        # Full system integration
│
├── 🖥️ app/                             # Gradio application (DermaScan AI)
│   ├── app.py                          # Main interface (~1 900 lines)
│   ├── model/final_model.pth           # Trained weights (gitignored → HF)
│   ├── rag/                            # FAISS index + knowledge base
│   ├── llm/                            # Qwen2.5-0.5B GGUF (gitignored → HF)
│   └── config/preprocessing_config.json
│
├── 📁 data/                            # Metadata CSVs + preprocessing configs
├── 📁 models/checkpoints/             # Training state JSONs
├── 📁 results/
│   ├── figures/                        # Training curves, confusion matrices, ROC
│   ├── metrics/                        # Per-class metrics, bootstrap CIs, history
│   └── tables/                         # LaTeX / PNG summary tables
│
├── .gitignore
├── requirements.txt
└── reproduce.md
```

---

## 🚀 Quick Start

### Run the Demo
The easiest way is the live demo on Hugging Face Spaces (no setup needed):

**[https://huggingface.co/spaces/graduationProject1/Skin-Cancer](https://huggingface.co/spaces/graduationProject1/Skin-Cancer)**

### Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/graduationp658-dotcom/Intelligent-Skin-Cancer-Diagnosis.git
cd Intelligent-Skin-Cancer-Diagnosis

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r app/requirements.txt

# 4. Download model weights from Hugging Face
#    Place final_model.pth → app/model/
#    Place qwen2.5-0.5b-instruct-q8_0.gguf → app/llm/
#    (Both files are hosted at the HF Space above)

# 5. Launch
python app/app.py
# Open http://127.0.0.1:7860
```

---

## 🧠 Model Architecture

```
Input image (224×224)
       │
EfficientNet-B4 backbone (ImageNet pretrained)
       │
Squeeze-and-Excitation block (channel attention, r=16)
       │
Global Average Pooling
       │
Classifier head:
  BatchNorm → Linear(1792→512) → ReLU → Dropout(0.4)
           → Linear(512→256)  → ReLU → Dropout(0.3)
           → Linear(256→8)
       │
Softmax → 8 class probabilities
```

Grad-CAM++ hooks are attached to the final convolutional block to produce class-specific localization maps.

---

## 🔬 Preprocessing Pipeline (v2)

Each input image passes through:

1. **Ruler removal** — border-threshold + inpainting
2. **Hair removal** — DullRazor-inspired black-hat morphology + inpainting
3. **Lesion segmentation** — Otsu thresholding + morphological cleanup
4. **Tight crop** — bounding box of the largest lesion contour (15% padding)
5. **Shadow removal** — LAB L-channel normalization
6. **CLAHE** — contrast-limited adaptive histogram equalization
7. **Reinhard color normalization** — LAB-space transfer to a reference distribution
8. **Background suppression** — soft Gaussian blur outside the lesion mask

---

## 🏥 Clinical RAG Pipeline

```
Question / predicted class
         │
Semantic search (all-MiniLM-L6-v2 + FAISS)
         │
Top-k relevant chunks retrieved (DermNet NZ + curated summaries)
         │
Qwen2.5-0.5B-Instruct (local GGUF, CPU)
         │
Grounded answer — friendly, conversational, no URLs, no hallucination
```

Knowledge base: DermNet NZ articles + project-authored curated summaries and ABCDE warning-sign descriptions for all 8 lesion classes.

---

## 📦 Dataset

- **Source**: HAM10000 (Human Against Machine with 10 000 training images) — ISIC Archive
- **Classes**: 8 lesion types (see above)
- **Preprocessing**: v2 pipeline applied to all images (see above)
- **Train / Val / Test split**: stratified by class

> Raw images and preprocessed datasets are not included in this repository due to size. Download HAM10000 from the [ISIC Archive](https://www.isic-archive.com/).

---

## ⚠️ Disclaimer

This system is an **educational and research prototype** developed as a graduation project. It is **NOT a medical device** and has **NOT been validated for clinical use**. All outputs must be reviewed by a qualified dermatologist or physician. Do not delay seeking medical care based on this tool's output.

---

## 📄 License

Code: [CC BY-NC 4.0](LICENSE) — non-commercial use only.  
Qwen2.5-0.5B model weights: [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).  
HAM10000 dataset: [CC BY-NC 4.0](https://www.isic-archive.com/terms-of-use).
