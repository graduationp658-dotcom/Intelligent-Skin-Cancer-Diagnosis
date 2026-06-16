---
title: DermaScan AI — Skin Lesion Classifier
emoji: 🔬
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# 🔬 DermaScan AI — Skin Lesion Classifier + Clinical Decision-Support

**Research / educational demo — NOT a medical device. NOT validated for clinical use.**

EfficientNet-B4 · Grad-CAM++ · Clinical RAG · Local LLM · Explainable AI

## What it does

### 🔬 Diagnosis tab
1. Upload a dermoscopy-style image (JPG/PNG).
2. EfficientNet-B4 + Squeeze-and-Excitation classifier predicts one of **8 lesion classes** (actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, squamous cell carcinoma, vascular lesion).
3. **Grad-CAM++** heatmap shows which region drove the prediction.
4. A structured **Diagnostic Support Report** is generated with urgency level, malignancy screening, clinical features, recommendation, and differential diagnoses — all grounded in the local knowledge base.
5. Ask follow-up questions about the result via the **AI assistant**.

### 💬 Clinical Assistant tab
Conversational Q&A about skin lesions, grounded **only** in the local clinical knowledge base (DermNet NZ + curated summaries). A local **Qwen2.5-0.5B-Instruct** LLM (GGUF, Apache 2.0) generates friendly, natural-language answers — no URLs sent to users, no external API calls.

### 📖 About & Safety tab
Full project description, model architecture, RAG pipeline, safety design, limitations, and ethical considerations.

## Files

| File | Description |
|------|-------------|
| `app.py` | Gradio Blocks interface (~1 900 lines) |
| `model/final_model.pth` | Trained classifier weights (~73 MB, Git LFS) |
| `rag/` | Pre-built FAISS index + knowledge base (offline) |
| `llm/qwen2.5-0.5b-instruct-q8_0.gguf` | Bundled local LLM (~644 MB, Git LFS, Apache 2.0) |
| `config/preprocessing_config.json` | Class list, image size, normalisation stats |

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:7860** in your browser.
