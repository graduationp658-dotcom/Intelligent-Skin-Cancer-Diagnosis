# Reproducibility Guide

This document describes how to reproduce the artifacts in this repository:
the trained model, evaluation metrics, Grad-CAM++ figures, the clinical RAG
report generator, and the Phase 8 tables/figures.

## 1. Environment

- Python 3.11 (tested on 3.11.0, Windows 11)
- NVIDIA GPU with CUDA 12.1 recommended (training + Grad-CAM++ use `cuda` if
  available, otherwise fall back to CPU)

```bash
python -m venv .venv
.venv\Scripts\activate                       # Windows
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`SEED = 42` is fixed for `numpy`, `torch`, and all `sklearn` splits/bootstraps
across every phase, so re-running the deterministic steps (splits, bootstrap
CIs, sample selection) reproduces the same numbers given the same input data
and model weights.

## 2. Raw data (not included in this repo)

| Source | Used for | Where to get it |
|---|---|---|
| HAM10000 | 7 of 8 classes | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) |
| ISIC Archive | additional squamous cell carcinoma images (8th class) | [ISIC Archive](https://www.isic-archive.com/) |

Expected layout at the project root:

```
HAM1000/
  HAM10000_images_part_1/
  HAM10000_images_part_2/
  HAM10000_metadata.csv
ISIC/
  Skin cancer ISIC The International Skin Imaging Collaboration/
```

## 3. Pipeline (run notebooks in order)

Each notebook should be run with **Restart Kernel & Run All** for a clean
reproduction. Outputs of each phase are consumed by later phases.

| Phase | Notebook | Produces |
|---|---|---|
| 1 | `skin_cancer_phase1.ipynb` | EDA (`results/metrics/eda_summary.csv`, `results/figures/eda_*`) |
| 2 | `skin_cancer_phase2.ipynb` | `data/preprocessing_config.json`, `data/{train,val,test}.csv`, preprocessed images |
| 3-4 | `skin_cancer_phase3.ipynb`, `skin_cancer_phase4.ipynb` | 3-stage progressive fine-tuning of EfficientNet-B4 + SE-block → `models/checkpoints/{best_model_stage1,2,3,final_model}.pth`, `results/metrics/{history_stage*,training_summary,model_architecture}.{csv,json}` |
| 5 | `skin_cancer_phase5.ipynb` | Test-set evaluation → `results/metrics/test_*.csv` (predictions, per-class metrics, bootstrap CIs, threshold tuning, target comparison) |
| 6 | `skin_cancer_phase6.ipynb` | Grad-CAM++ implementation + example visualizations → `results/figures/gradcam_examples.png` |
| 7 | `skin_cancer_phase7_clinical_rag.ipynb` | Clinical knowledge base + FAISS vector store (`results/rag/`) and `predict_and_explain()` |
| 8 | `skin_cancer_phase8.ipynb` | Final report tables (`results/tables/`) and figures (`results/figures/figure*.png`), this document |

Phases 3-4 are the only computationally expensive step (GPU training, ~tens
of minutes per stage on an RTX 2000 Ada). All other phases run in at most a
few minutes and only *read* the artifacts listed above -- Phase 8 in
particular performs no retraining.

Phase 7's knowledge-base construction scrapes DermNet NZ pages over the
network with curated fallback content if a page is unreachable; the
already-built `results/rag/knowledge_base.json`, `chunk_metadata.json`, and
`faiss_index.bin` are checked into `results/rag/` so Phase 7 (and the Gradio
app) can run fully offline using the cached knowledge base.

## 4. Key model facts

- Architecture: EfficientNet-B4 backbone (`torchvision.models.efficientnet_b4`,
  `weights=None`) + a Squeeze-and-Excitation block (1792 channels,
  reduction=16) before the classifier head.
- Input: 224x224 RGB, normalized with ImageNet mean/std
  (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`).
- 8 classes (alphabetical): actinic keratosis, basal cell carcinoma,
  dermatofibroma, melanoma, nevus, pigmented benign keratosis, squamous cell
  carcinoma, vascular lesion.
- Final checkpoint: `models/checkpoints/final_model.pth`
  (val macro AUC = 0.8550, test macro AUC = 0.8734).

## 5. Gradio demo / deployment

See `app/README.md` for the Gradio interface (image upload → prediction,
Grad-CAM++ overlay, clinical RAG report) and Hugging Face Spaces deployment
instructions.
