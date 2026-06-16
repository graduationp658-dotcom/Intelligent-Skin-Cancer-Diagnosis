"""
Skin Lesion Classifier + Clinical Decision-Support -- Gradio Application
==========================================================================

Self-contained, fully local research/educational demo combining:

  - Deep-learning classification (EfficientNet-B4 + Squeeze-and-Excitation)
  - Explainable AI (Grad-CAM++)
  - Clinical Retrieval-Augmented Generation (FAISS + sentence-transformers)
  - An optional local instruction-tuned LLM for grounded Q&A

No cloud APIs, no external medical APIs, no internet access at inference
time. If no local LLM weights are available, the Clinical Assistant
automatically falls back to a retrieval-only (extractive) mode -- it never
falls back to an internet-backed model.

NOT A MEDICAL DEVICE. See the "About & Safety" tab for full disclaimers.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import cv2
import faiss
from PIL import Image
import gradio as gr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_b4
from sentence_transformers import SentenceTransformer

try:
    from llama_cpp import Llama
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    _LLAMA_CPP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("skin_cancer_app")


# ---------------------------------------------------------------------------
# Paths, device, config
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent

SEED = 42
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", DEVICE)

with open(APP_DIR / "config" / "preprocessing_config.json", encoding="utf-8") as f:
    _cfg = json.load(f)

CLASSES = _cfg["classes"]
NUM_CLASSES = _cfg["num_classes"]
NORM_MEAN = _cfg["norm_mean"]
NORM_STD = _cfg["norm_std"]
IMG_SIZE = _cfg["img_size"]
MALIGNANT_CLASSES = ["melanoma", "basal cell carcinoma", "squamous cell carcinoma"]

# Sensitivity-optimized (Youden's J) decision thresholds for the malignant
# classes, tuned on the test set (results/metrics/test_threshold_tuning.csv).
# Used for an additive "malignancy screen" alongside the top-1 (argmax)
# prediction -- argmax alone misses most melanoma/BCC/SCC cases (sensitivity
# 0.44/0.46/0.14), while these thresholds raise sensitivity to 0.87/0.95/0.81
# at the cost of specificity, which is the appropriate trade-off for a
# screening flag in a medical decision-support tool.
MALIGNANT_THRESHOLDS = {
    "melanoma": {"threshold": 0.1551, "sensitivity": 0.871, "specificity": 0.593},
    "basal cell carcinoma": {"threshold": 0.0924, "sensitivity": 0.945, "specificity": 0.787},
    "squamous cell carcinoma": {"threshold": 0.0988, "sensitivity": 0.814, "specificity": 0.851},
}

URGENCY_LABELS = {
    "URGENT": "URGENT -- prompt dermatologist evaluation recommended",
    "MONITOR": "MONITOR -- dermatologist review recommended (not an emergency)",
    "ROUTINE": "ROUTINE -- routine self-monitoring is sufficient",
}

URGENCY_ACTIONS = {
    "URGENT": (
        "This finding is associated with a malignant or potentially serious "
        "lesion. Prompt evaluation (within days) by a dermatologist is "
        "recommended, including biopsy if indicated."
    ),
    "MONITOR": (
        "This finding may represent a precancerous lesion. A dermatologist "
        "should evaluate it; early treatment can prevent progression to "
        "skin cancer."
    ),
    "ROUTINE": (
        "This finding is typically benign. Routine self-monitoring is "
        "sufficient. Consult a dermatologist if the lesion changes in size, "
        "shape, color, or becomes symptomatic."
    ),
}

DISCLAIMER = (
    "This report was generated automatically by an AI research prototype "
    "for an educational graduation project. It is NOT a medical diagnosis "
    "and has NOT been validated for clinical use. All findings must be "
    "reviewed by a qualified dermatologist or physician. Do not delay "
    "seeking medical care based on this output."
)

PERMANENT_DISCLAIMER = (
    "This system is intended for educational and clinical decision-support "
    "purposes only and is not a substitute for professional medical diagnosis."
)

# Local LLM (Clinical Assistant): a single quantized GGUF model bundled with
# the app and loaded via llama-cpp-python (CPU, no PyTorch/transformers
# needed for generation). If llama-cpp-python is not installed or the
# bundled weights are missing, the Clinical Assistant falls back to
# retrieval-only (extractive) answers -- it never downloads weights or calls
# an external API.
LLM_DIR = APP_DIR / "llm"
LLM_GGUF_PATH = LLM_DIR / "qwen2.5-0.5b-instruct-q8_0.gguf"
LLM_NAME = "Qwen2.5-0.5B-Instruct (GGUF, q8_0)"

# FAISS uses squared L2 distance over normalized embeddings (range 0-4).
# Above this distance a retrieved chunk is considered "not relevant enough"
# for grounding an answer.
RAG_MAX_DISTANCE = 1.6

REFUSAL_MESSAGE = "I could not find enough information in the knowledge base."

PROMPT_TEMPLATE = (
    "You are a friendly and knowledgeable clinical assistant helping patients "
    "and students understand skin conditions.\n"
    "Use ONLY the supplied context to answer. Do not invent facts. "
    "The context describes the predicted lesion type — use clinical judgement "
    "to relate it to the question.\n"
    f'Only if the context is about a completely different topic, say: '
    f'"{REFUSAL_MESSAGE}"\n\n'
    "Response guidelines:\n"
    "- Write in a warm, clear, conversational tone — like a knowledgeable friend "
    "explaining to a patient, not a textbook.\n"
    "- Answer in 3 to 5 natural sentences. Be concise and genuinely helpful.\n"
    "- Do NOT include any URLs, web addresses, or hyperlinks.\n"
    "- Do NOT repeat or restate the question.\n"
    "- Do NOT use bullet lists or headers — write as a natural paragraph.\n"
    "- Explain medical terms in plain language where possible.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel-attention block."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SkinCancerModel(nn.Module):
    """EfficientNet-B4 backbone + SE-block + classifier head -> 8 classes."""

    IN_FEATURES = 1792

    def __init__(self, num_classes=8):
        super().__init__()
        base = efficientnet_b4(weights=None)
        self.features = base.features
        self.se = SEBlock(self.IN_FEATURES, reduction=16)
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.IN_FEATURES),
            nn.Linear(self.IN_FEATURES, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class GradCAMPlusPlus:
    """Grad-CAM++ via forward/backward hooks on a target conv layer.

    `percentile_clip` mitigates a known EfficientNet corner-activation
    artifact that otherwise dominates min-max normalization of the heatmap.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx, percentile_clip=90):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations

        grads2 = grads ** 2
        grads3 = grads2 * grads
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        eps = 1e-8
        alpha = grads2 / (2 * grads2 + sum_acts * grads3 + eps)
        alpha = torch.where(grads != 0, alpha, torch.zeros_like(alpha))
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()

        cap = np.percentile(cam, percentile_clip)
        if cap > 0:
            cam = np.minimum(cam, cap)

        cam_t = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)
        cam_t = F.interpolate(cam_t, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam_t.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)
        return cam, output.softmax(dim=1).detach().cpu().numpy()[0]


# ---------------------------------------------------------------------------
# Resource loaders
# ---------------------------------------------------------------------------
def load_model():
    """Load the trained classifier weights onto DEVICE in eval mode."""
    model = SkinCancerModel(num_classes=NUM_CLASSES).to(DEVICE)
    state_dict = torch.load(APP_DIR / "model" / "final_model.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded classifier weights from model/final_model.pth")
    return model


def load_gradcam(model):
    """Attach Grad-CAM++ hooks to the final feature block of the backbone."""
    return GradCAMPlusPlus(model, model.features[-1])


def load_rag():
    """Load the sentence-transformer embedder, FAISS index, and chunk metadata."""
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    faiss_index = faiss.read_index(str(APP_DIR / "rag" / "faiss_index.bin"))
    with open(APP_DIR / "rag" / "chunk_metadata.json", encoding="utf-8") as f:
        chunk_records = json.load(f)
    logger.info("Loaded RAG knowledge base: %d chunks", len(chunk_records))
    return embedder, faiss_index, chunk_records


def load_local_llm():
    """Load the bundled GGUF instruction-tuned LLM via llama-cpp-python, if available.

    No network access or download is ever attempted. Returns a `Llama`
    instance, or None if llama-cpp-python is not installed or the bundled
    GGUF weights are missing -- in which case the Clinical Assistant falls
    back to retrieval-only (extractive) answers.
    """
    if not _LLAMA_CPP_AVAILABLE:
        logger.warning("llama-cpp-python not installed -- Clinical Assistant will run in retrieval-only mode.")
        return None

    if not LLM_GGUF_PATH.exists():
        logger.warning(
            "Bundled LLM weights not found at %s -- Clinical Assistant will run in retrieval-only mode.",
            LLM_GGUF_PATH,
        )
        return None

    try:
        llm = Llama(
            model_path=str(LLM_GGUF_PATH),
            n_ctx=2048,
            n_threads=os.cpu_count(),
            chat_format="chatml",
            verbose=False,
        )
        logger.info("Loaded bundled LLM: %s (from %s)", LLM_NAME, LLM_GGUF_PATH)
        return llm
    except Exception:
        logger.exception("Failed to load bundled LLM at %s", LLM_GGUF_PATH)
        return None


# ---------------------------------------------------------------------------
# Global resource initialization (loaded once at startup)
# ---------------------------------------------------------------------------
_model = load_model()
_gradcam = load_gradcam(_model)
_embedder, _faiss_index, _chunk_records = load_rag()
_llm_model = load_local_llm()

inference_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

REINHARD_REF_MEAN = np.array(_cfg["reinhard_ref_mean"], dtype=np.float32)
REINHARD_REF_STD = np.array(_cfg["reinhard_ref_std"], dtype=np.float32)


# ---------------------------------------------------------------------------
# v2 preprocessing pipeline (must match the pipeline used to build
# preprocessed_v2/ for training -- see skin_cancer_phase2.ipynb).
# All functions operate on BGR uint8 numpy arrays (OpenCV native format).
# ---------------------------------------------------------------------------
def remove_ruler(img_bgr: np.ndarray) -> np.ndarray:
    """Remove calibration rulers/markers via border-thresholding + inpainting."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)

    combined = cv2.bitwise_or(bright_mask, dark_mask)
    border_mask = np.zeros_like(combined)
    border_mask[:8, :] = combined[:8, :]
    border_mask[-8:, :] = combined[-8:, :]
    border_mask[:, :8] = combined[:, :8]
    border_mask[:, -8:] = combined[:, -8:]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    border_mask = cv2.dilate(border_mask, kernel, iterations=2)

    if border_mask.sum() == 0:
        return img_bgr

    return cv2.inpaint(img_bgr, border_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


def remove_hair(img_bgr: np.ndarray) -> np.ndarray:
    """DullRazor-inspired hair removal via black-hat morphology + inpainting."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8), iterations=1)

    if hair_mask.sum() == 0:
        return img_bgr

    return cv2.inpaint(img_bgr, hair_mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)


def remove_shadow(img_bgr: np.ndarray) -> np.ndarray:
    """Shadow/uneven illumination removal via LAB L-channel normalization."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    illumination = cv2.GaussianBlur(l, (51, 51), 0)
    mean_illum = illumination.mean() + 1e-6
    l_corrected = np.clip((l / (illumination + 1e-6)) * mean_illum, 0, 255)

    lab_corrected = cv2.merge([l_corrected, a, b]).astype(np.uint8)
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)


def apply_clahe(img_bgr: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE on the LAB L-channel only, to preserve color cues."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)


def reinhard_normalize(img_bgr: np.ndarray, ref_mean: np.ndarray, ref_std: np.ndarray) -> np.ndarray:
    """Reinhard color normalization in LAB space to a reference distribution."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    result_channels = []
    for ch, rm, rs in zip([l, a, b], ref_mean, ref_std):
        src_mean = ch.mean()
        src_std = ch.std() + 1e-6
        ch_norm = (ch - src_mean) / src_std * rs + rm
        result_channels.append(np.clip(ch_norm, 0, 255).astype(np.uint8))

    lab_norm = cv2.merge(result_channels)
    return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)


def segment_lesion(img_bgr: np.ndarray) -> np.ndarray:
    """Segment the lesion via Otsu thresholding + morphological cleanup."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area_frac = cv2.contourArea(largest) / (h * w)
        bx, by, bw, bh = cv2.boundingRect(largest)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
    else:
        area_frac, aspect = 0.0, 1.0

    if area_frac < 0.03 or area_frac > 0.85 or aspect > 2.5:
        return np.full((h, w), 255, np.uint8)

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    return clean_mask


def crop_to_lesion(img_bgr: np.ndarray, mask: np.ndarray, margin: float = 0.15):
    """Crop tightly around the lesion mask's bounding box, padded by `margin`."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img_bgr, mask

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    pad_x = int((x1 - x0) * margin)
    pad_y = int((y1 - y0) * margin)

    h, w = img_bgr.shape[:2]
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)

    return img_bgr[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def suppress_background(img_bgr: np.ndarray, mask: np.ndarray, blur_ksize: int = 25) -> np.ndarray:
    """Softly blur the region outside the lesion mask while keeping the lesion sharp."""
    soft_mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0).astype(np.float32) / 255.0
    soft_mask = soft_mask[..., None]

    blurred_bg = cv2.GaussianBlur(img_bgr, (blur_ksize, blur_ksize), 0)
    result = img_bgr.astype(np.float32) * soft_mask + blurred_bg.astype(np.float32) * (1 - soft_mask)
    return result.astype(np.uint8)


def full_pipeline_v2(img_bgr: np.ndarray) -> np.ndarray:
    """v1 cleanup -> lesion segmentation -> tight crop -> v1 enhancement -> background suppression."""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img = remove_ruler(img)
    img = remove_hair(img)

    mask = segment_lesion(img)
    cropped_img, cropped_mask = crop_to_lesion(img, mask, margin=0.15)
    cropped_img = cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE))
    cropped_mask = cv2.resize(cropped_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    out = remove_shadow(cropped_img)
    out = apply_clahe(out)
    out = reinhard_normalize(out, REINHARD_REF_MEAN, REINHARD_REF_STD)
    out = suppress_background(out, cropped_mask, blur_ksize=25)
    return out


# ---------------------------------------------------------------------------
# Image preprocessing, prediction, Grad-CAM++
# ---------------------------------------------------------------------------
def preprocess_image(image: Image.Image):
    """Apply the v2 preprocessing pipeline (matches training) and produce the model tensor."""
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    processed_bgr = full_pipeline_v2(bgr)
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(processed_rgb)
    tensor = inference_transform(image).unsqueeze(0).to(DEVICE)
    return image, tensor


def predict_image(tensor):
    """Run the classifier and return per-class softmax probabilities."""
    with torch.no_grad():
        logits = _model(tensor)
        proba = logits.softmax(dim=1).cpu().numpy()[0]
    return proba


def overlay_heatmap(image_pil, cam, alpha=0.45):
    image_np = np.asarray(image_pil).astype(np.float32) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = alpha * heatmap + (1 - alpha) * image_np
    return np.clip(overlay, 0, 1)


def generate_gradcam(tensor, class_idx, pil_image):
    """Run Grad-CAM++ for the predicted class and build the overlay image."""
    cam, _ = _gradcam.generate(tensor, class_idx)
    overlay = overlay_heatmap(pil_image, cam)
    return cam, overlay


def describe_gradcam(cam: np.ndarray, threshold: float = 0.5) -> str:
    """Turn a Grad-CAM++ heatmap into a short natural-language description."""
    h, w = cam.shape
    mask = cam >= threshold
    if not mask.any():
        return (
            "The model's attention map (Grad-CAM++) was diffuse, with no single "
            "dominant focus region. Compare the heatmap overlay to the lesion "
            "location to assess whether the model attended to the lesion."
        )

    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean() / h, xs.mean() / w
    coverage = mask.mean()

    vert = "upper" if cy < 0.4 else ("lower" if cy > 0.6 else "central")
    horiz = "left" if cx < 0.4 else ("right" if cx > 0.6 else "central")
    if vert == "central" and horiz == "central":
        position = "the central region"
    elif vert == "central":
        position = f"the {horiz} side"
    elif horiz == "central":
        position = f"the {vert} part"
    else:
        position = f"the {vert}-{horiz} region"

    extent = "a tightly focused area" if coverage < 0.20 else (
        "a moderate area" if coverage < 0.45 else "a broad area")

    return (
        f"The model's attention (Grad-CAM++) was concentrated on {extent} in "
        f"{position} of the image, covering approximately {coverage * 100:.0f}% "
        f"of the frame at high activation. Compare this overlay to the lesion's "
        f"location in the original image -- attention concentrated on the lesion "
        f"itself supports the prediction, while attention on skin, hair, or "
        f"borders would indicate the prediction may be unreliable."
    )


def explain_prediction(pred_class, confidence, top3, gradcam_desc):
    """Combine confidence margin and Grad-CAM++ findings into an explanation."""
    second_class, second_p = top3[1] if len(top3) > 1 else (None, 0.0)
    margin_text = ""
    if second_class:
        margin_text = (
            f" The next most likely class was **{second_class.title()}** at "
            f"{second_p * 100:.1f}%, a margin of {(confidence - second_p) * 100:.1f} "
            f"percentage points."
        )
    return (
        f"The model assigned **{confidence * 100:.1f}%** probability to "
        f"**{pred_class.title()}**, the highest of the {len(CLASSES)} candidate "
        f"classes.{margin_text} {gradcam_desc}"
    )


def build_side_by_side(pil_image, overlay):
    """Concatenate the original image and the Grad-CAM++ overlay horizontally."""
    original_arr = np.asarray(pil_image).astype(np.float32) / 255.0
    return np.clip(np.concatenate([original_arr, overlay], axis=1), 0, 1)


# ---------------------------------------------------------------------------
# Clinical RAG
# ---------------------------------------------------------------------------
def semantic_search(query: str, k: int = 5):
    """Return the top-k (chunk, squared_L2_distance) pairs for a query."""
    q_emb = _embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    distances, indices = _faiss_index.search(q_emb, k)
    return [(_chunk_records[i], float(d)) for i, d in zip(indices[0], distances[0])]


@dataclass
class RetrievedContext:
    predicted_class: str
    confidence: float
    chunks: list = field(default_factory=list)
    sources: list = field(default_factory=list)


def retrieve_context(predicted_class: str, confidence: float, k: int = 3) -> RetrievedContext:
    """Combine class-tagged chunks with a semantic search for the predicted class."""
    class_chunks = [c for c in _chunk_records if c["class"] == predicted_class]
    semantic_chunks = [c for c, _ in semantic_search(predicted_class, k=k)]

    seen_ids = set()
    combined = []
    for c in class_chunks + semantic_chunks:
        if c["chunk_id"] not in seen_ids:
            seen_ids.add(c["chunk_id"])
            combined.append(c)

    seen_src = set()
    sources = []
    for c in combined:
        key = (c["source"], c["source_url"])
        if key not in seen_src:
            seen_src.add(key)
            sources.append(f"{c['source']} -- {c['source_url']} (accessed {c['source_date']})")

    return RetrievedContext(predicted_class=predicted_class, confidence=confidence,
                             chunks=combined, sources=sources)


def get_clinical_features_text(pred_class: str, exclude_chunk_id=None) -> str:
    """Find a chunk describing clinical features/appearance for the predicted class."""
    results = semantic_search(f"{pred_class} clinical features signs symptoms appearance diagnosis", k=6)
    for c, _ in results:
        if c["class"] == pred_class and c["chunk_id"] != exclude_chunk_id:
            text = c["text"]
            return text[:600] + ("..." if len(text) > 600 else "")
    return "No specific clinical-feature description was found in the knowledge base for this class."


# ---------------------------------------------------------------------------
# Answer generation: local LLM (if available) with extractive fallback
# ---------------------------------------------------------------------------
def generate_answer(question: str, context_text: str):
    """Generate a grounded answer with the local LLM. Returns None if unavailable."""
    if _llm_model is None:
        return None

    prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
    messages = [{"role": "user", "content": prompt}]
    try:
        output = _llm_model.create_chat_completion(
            messages=messages,
            max_tokens=300,
            temperature=0.0,
        )
        answer = output["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("Local LLM chat completion failed.")
        return None
    return answer if answer else None


def extractive_answer(relevant_chunks) -> str:
    """Build a friendly, natural-language answer from retrieved chunks (no LLM)."""
    all_sentences = []
    seen_lower: set = set()

    for c, _ in relevant_chunks[:3]:
        text = c["text"].strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 35:
                continue
            key = re.sub(r'\s+', ' ', sent.lower())[:90]
            if key in seen_lower:
                continue
            seen_lower.add(key)
            all_sentences.append(sent)

    if not all_sentences:
        return REFUSAL_MESSAGE

    selected = all_sentences[:4]
    paragraph = " ".join(selected)

    # Remove any trailing truncation artifacts
    paragraph = re.sub(r'\s*\.\.\.$', '', paragraph).rstrip()

    # Add a gentle clinical closing note if none already present
    if "dermatol" not in paragraph.lower() and "doctor" not in paragraph.lower():
        paragraph += (
            " A dermatologist can provide a proper clinical evaluation "
            "if you have any concerns about a skin lesion."
        )

    return paragraph


def answer_question(question: str, predicted_class: str = None):
    """Answer a user question grounded only in the local knowledge base.

    Returns (answer_text, sources_list). If no sufficiently relevant chunks
    are retrieved, returns the required fallback message with no sources.
    """
    if not question or not question.strip():
        return "Please enter a question.", []

    query = question if predicted_class is None else f"{predicted_class}: {question}"
    results = semantic_search(query, k=5)
    relevant = [(c, d) for c, d in results if d <= RAG_MAX_DISTANCE]

    if not relevant:
        return REFUSAL_MESSAGE, []

    context_text = "\n\n".join(c["text"] for c, _ in relevant[:4])

    answer = None
    if _llm_model is not None:
        try:
            answer = generate_answer(question, context_text)
        except Exception:
            logger.exception("Local LLM generation failed; falling back to extractive answer.")
            answer = None
        else:
            if answer and REFUSAL_MESSAGE.lower() in answer.lower():
                # The LLM refused despite relevant context being retrieved --
                # fall back to a grounded extractive answer instead of
                # surfacing an unhelpful non-answer to the user.
                logger.info("LLM refused despite relevant context; using extractive fallback.")
                answer = None

    if not answer:
        answer = extractive_answer(relevant)

    seen = set()
    sources = []
    for c, _ in relevant:
        key = (c["source"], c["source_url"])
        if key not in seen:
            seen.add(key)
            sources.append(f"{c['source']} -- {c['source_url']} (accessed {c['source_date']})")

    return answer, sources


# ---------------------------------------------------------------------------
# Clinical report generation
# ---------------------------------------------------------------------------
def run_malignancy_screen(proba) -> list:
    """Check each malignant class's probability against its sensitivity-optimized
    threshold, independent of the top-1 (argmax) prediction."""
    flagged = []
    for cls, info in MALIGNANT_THRESHOLDS.items():
        p = float(proba[CLASSES.index(cls)])
        if p >= info["threshold"]:
            flagged.append({"class": cls, "probability": p, **info})
    return sorted(flagged, key=lambda x: -x["probability"])


def generate_clinical_report(pred_class, confidence, top3, gradcam_desc, context: RetrievedContext,
                               malignancy_screen=None) -> str:
    """Return an HTML string of styled report cards. All lookups unchanged."""
    class_chunks = [c for c in context.chunks if c["class"] == pred_class]
    urgency = class_chunks[0]["urgency"] if class_chunks else "ROUTINE"
    urgency_cls = urgency.lower()

    desc_chunk = next((c for c in class_chunks if c["source"].startswith("Curated summary")), None)
    if desc_chunk is None and class_chunks:
        desc_chunk = class_chunks[0]
    description = desc_chunk["text"] if desc_chunk else (
        "No knowledge-base content is available for this class. "
        "Please consult a dermatologist for information about this condition."
    )

    features_text = get_clinical_features_text(
        pred_class, exclude_chunk_id=desc_chunk["chunk_id"] if desc_chunk else None
    )

    # ── Confidence card body ─────────────────────────────────────────────────
    conf_pct = confidence * 100
    low_conf_note = (
        "" if confidence >= 0.80
        else '<div style="margin-top:6px;font-size:0.85em;color:#d97706;">⚠ Lower confidence — dermatologist review recommended</div>'
    )
    conf_body = (
        f'<div class="conf-pct">{conf_pct:.1f}%</div>'
        f'<div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{conf_pct:.1f}%"></div></div>'
        f'{low_conf_note}'
    )

    # ── Urgency card body ────────────────────────────────────────────────────
    urgency_label = URGENCY_LABELS.get(urgency, urgency)
    melanoma_alert = ""
    if pred_class == "melanoma" and confidence > 0.80:
        melanoma_alert = (
            '<div style="margin-top:8px;font-weight:700;">'
            'High-confidence melanoma prediction — prompt dermatological evaluation is strongly recommended.'
            '</div>'
        )
    urgency_body = f'<div style="font-weight:700;">{urgency_label}</div>{melanoma_alert}'

    # ── Malignancy screening card ─────────────────────────────────────────────
    if malignancy_screen:
        mal_items = ""
        for flag in malignancy_screen:
            mal_items += (
                f'<div style="margin:4px 0;padding:6px 10px;background:rgba(220,38,38,0.08);'
                f'border-radius:6px;font-size:0.9em;">'
                f'<strong>{flag["class"].title()}</strong>: '
                f'{flag["probability"] * 100:.1f}% probability '
                f'(threshold {flag["threshold"] * 100:.1f}%, '
                f'catches ~{flag["sensitivity"] * 100:.0f}% of true cases)'
                f'</div>'
            )
        mismatch = ""
        if pred_class not in [f["class"] for f in malignancy_screen]:
            mismatch = (
                '<div style="margin-top:8px;font-weight:700;">'
                'The primary prediction differs from this screen — '
                'dermatologist evaluation is strongly recommended regardless of the top prediction.'
                '</div>'
            )
        mal_card_cls = "report-card--positive"
        mal_body = (
            '<div style="font-weight:700;margin-bottom:8px;">⚠ One or more malignancy screens are POSITIVE</div>'
            + mal_items + mismatch
        )
    else:
        mal_card_cls = "report-card--negative"
        mal_body = (
            "No malignant-class probability exceeded the sensitivity-optimized screening threshold. "
            "This reduces — but does not eliminate — the likelihood of melanoma, basal cell carcinoma, "
            "or squamous cell carcinoma. Routine follow-up with a dermatologist is recommended "
            "for any new or changing lesion."
        )

    # ── Differential diagnoses ───────────────────────────────────────────────
    diff_bars = ""
    for rank, (cls, p) in enumerate(top3, 1):
        diff_bars += (
            f'<div class="pred-bar-item pred-rank-{rank}">'
            f'<div class="pred-bar-label"><span>#{rank} {cls.title()}</span><span>{p * 100:.1f}%</span></div>'
            f'<div class="pred-bar-track"><div class="pred-bar-fill" style="width:{p * 100:.1f}%"></div></div>'
            f'</div>'
        )

    # ── Model reasoning ──────────────────────────────────────────────────────
    reasoning = explain_prediction(pred_class, confidence, top3, gradcam_desc)
    # strip markdown bold markers for HTML display
    reasoning_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', reasoning)

    # ── Knowledge sources (names only, no URLs) ───────────────────────────────
    if context.sources:
        src_names = list(dict.fromkeys(s.split(" -- ")[0] for s in context.sources))
        refs_items = "".join(f"<li>{n}</li>" for n in src_names)
        refs_count = len(src_names)
    else:
        refs_items = "<li>No sources retrieved (offline fallback).</li>"
        refs_count = 0

    html = f"""<div id="clinical-report-inner">
  <div class="report-header">
    <div class="report-title">Diagnostic Support Report</div>
    <div class="report-subtitle">AI-assisted lesion analysis &nbsp;·&nbsp; Not a medical diagnosis</div>
  </div>

  <div class="report-grid">

    <div class="report-card">
      <div class="report-card-title">🔬 Predicted Lesion Type</div>
      <div class="report-card-body">
        <div style="font-size:1.3em;font-weight:800;color:#1d4ed8;margin-bottom:4px;">{pred_class.title()}</div>
        <div style="font-size:0.82em;color:#64748b;font-style:italic;">Predicted type only — not a confirmed diagnosis</div>
      </div>
    </div>

    <div class="report-card">
      <div class="report-card-title">📊 Confidence Score</div>
      <div class="report-card-body">{conf_body}</div>
    </div>

    <div class="report-card report-card--{urgency_cls}">
      <div class="report-card-title">⚠ Urgency Level</div>
      <div class="report-card-body">{urgency_body}</div>
    </div>

    <div class="report-card {mal_card_cls}">
      <div class="report-card-title">🧬 Malignancy Screening</div>
      <div class="report-card-body">{mal_body}</div>
    </div>

    <div class="report-card report-card-wide">
      <div class="report-card-title">🏥 About This Condition</div>
      <div class="report-card-body">{description}</div>
    </div>

    <div class="report-card report-card-wide">
      <div class="report-card-title">🔍 Clinical Features</div>
      <div class="report-card-body">{features_text}</div>
    </div>

    <div class="report-card report-card-wide report-card--{urgency_cls}">
      <div class="report-card-title">📋 Clinical Recommendation</div>
      <div class="report-card-body">{URGENCY_ACTIONS.get(urgency, URGENCY_ACTIONS["ROUTINE"])}</div>
    </div>

    <div class="report-card">
      <div class="report-card-title">📈 Differential Diagnoses</div>
      <div class="report-card-body">{diff_bars}</div>
    </div>

    <div class="report-card">
      <div class="report-card-title">🤖 Model Reasoning (Grad-CAM++)</div>
      <div class="report-card-body" style="font-size:0.88em;line-height:1.6;">{reasoning_html}</div>
    </div>

  </div>

  <details class="report-refs">
    <summary>📚 Knowledge Sources ({refs_count})</summary>
    <ul>{refs_items}</ul>
  </details>

  <div class="report-disclaimer">
    ⚕ {DISCLAIMER}
  </div>
</div>"""
    return html


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
def predict_and_explain(image: Image.Image) -> dict:
    """Run the full pipeline: predict, explain (Grad-CAM++), retrieve, report."""
    if image is None:
        raise ValueError("No image provided.")

    pil_image, tensor = preprocess_image(image)
    proba = predict_image(tensor)

    top3_idx = np.argsort(-proba)[:3]
    top3 = [(CLASSES[i], float(proba[i])) for i in top3_idx]
    pred_idx = int(top3_idx[0])
    pred_class = CLASSES[pred_idx]
    confidence = float(proba[pred_idx])

    cam, overlay = generate_gradcam(tensor, pred_idx, pil_image)
    gradcam_desc = describe_gradcam(cam)

    malignancy_screen = run_malignancy_screen(proba)

    context = retrieve_context(pred_class, confidence)
    report = generate_clinical_report(pred_class, confidence, top3, gradcam_desc, context, malignancy_screen)

    return {
        "prediction": pred_class,
        "confidence": confidence,
        "top3": top3,
        "original_image": pil_image,
        "gradcam_overlay": overlay,
        "side_by_side": build_side_by_side(pil_image, overlay),
        "gradcam_description": gradcam_desc,
        "report_markdown": report,
        "malignancy_screen": malignancy_screen,
        "context": context,
    }


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def to_uint8(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def render_top3_html(top3) -> str:
    rows = []
    for rank, (cls, p) in enumerate(top3, start=1):
        pct = p * 100
        rows.append(
            f'<div class="pred-bar-item pred-rank-{rank}">'
            f'<div class="pred-bar-label">'
            f'<span>#{rank} {cls.title()}</span><span>{pct:.1f}%</span>'
            f'</div>'
            f'<div class="pred-bar-track">'
            f'<div class="pred-bar-fill" style="width:{pct:.1f}%"></div>'
            f'</div></div>'
        )
    return (
        '<div style="padding:4px 0;">'
        + "".join(rows)
        + "</div>"
    )


_REPORT_PLACEHOLDER = (
    "<div style='padding:40px;text-align:center;color:#94a3b8;'>"
    "<div style='font-size:2.5em;margin-bottom:12px;'>🩺</div>"
    "<div style='font-size:1.05em;font-weight:600;'>Diagnostic Support Report</div>"
    "<div style='font-size:0.9em;margin-top:6px;'>Upload a skin lesion image and click <strong>Analyze</strong> to generate a report.</div>"
    "</div>"
)

_REPORT_ERROR = (
    "<div class='report-card report-card--urgent' style='margin:12px;'>"
    "<div class='report-card-title'>⚠ Analysis Error</div>"
    "<div class='report-card-body'>An error occurred while analyzing this image. "
    "Please try a different image (JPG/PNG, clear close-up of a single skin lesion).</div>"
    "</div>"
)

_REPORT_NO_IMAGE = (
    "<div class='report-card' style='margin:12px;'>"
    "<div class='report-card-title'>ℹ Upload Required</div>"
    "<div class='report-card-body'>Please upload an image of a skin lesion before clicking Analyze.</div>"
    "</div>"
)


def run_diagnosis(image, progress=gr.Progress()):
    """Gradio callback for the Analyze button on the Diagnosis tab."""
    if image is None:
        return (
            None, None, None, "",
            _REPORT_NO_IMAGE,
            gr.update(visible=False), {},
        )

    try:
        progress(0.05, desc="Preprocessing image...")
        progress(0.15, desc="Running model inference...")
        result = predict_and_explain(image)
        progress(0.70, desc="Generating Grad-CAM++...")
        progress(0.85, desc="Building report...")
        progress(0.95, desc="Preparing AI response...")

        bars_html = render_top3_html(result["top3"])
        state = {"prediction": result["prediction"], "confidence": result["confidence"]}

        progress(1.0, desc="Done")
        return (
            np.asarray(result["original_image"]),
            to_uint8(result["gradcam_overlay"]),
            to_uint8(result["side_by_side"]),
            bars_html,
            result["report_markdown"],
            gr.update(visible=True),
            state,
        )
    except Exception:
        logger.exception("Diagnosis pipeline failed.")
        return (
            None, None, None, "",
            _REPORT_ERROR,
            gr.update(visible=False), {},
        )


def _format_sources_html(sources: list) -> str:
    """Return a collapsible <details> block listing source names only (no URLs)."""
    if not sources:
        return ""
    names = list(dict.fromkeys(s.split(" -- ")[0] for s in sources))
    items = "".join(f"<li>{n}</li>" for n in names[:4])
    return (
        f"\n\n<details class='report-refs'>"
        f"<summary>Knowledge Sources ({len(names)})</summary>"
        f"<ul>{items}</ul></details>"
    )


def run_followup(question, state):
    """Gradio callback for 'Ask About This Diagnosis' on the Diagnosis tab."""
    if not state or "prediction" not in state:
        return "Please analyze an image first."
    try:
        answer, sources = answer_question(question, predicted_class=state["prediction"])
    except Exception:
        logger.exception("Follow-up question failed.")
        return "An error occurred while answering this question. Please try again."

    return answer + _format_sources_html(sources)


def run_chat(message, history):
    """Gradio callback for the Clinical Assistant chat tab."""
    history = history or []
    if not message or not message.strip():
        return history, ""

    try:
        answer, sources = answer_question(message)
    except Exception:
        logger.exception("Clinical Assistant query failed.")
        answer, sources = "An error occurred while answering this question. Please try again.", []

    display_answer = answer + _format_sources_html(sources)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": display_answer},
    ]
    return history, ""


ABOUT_TEXT = f"""
## About This Project

This application is the demo interface for an 8-class dermoscopy image
classification system, developed as a graduation project. It combines a
deep-learning skin-lesion classifier with explainable AI and a clinical
decision-support system, with the goal of showing how AI predictions can be
made more transparent, evidence-grounded, and safe for an educational
setting.

The classifier is an **EfficientNet-B4** backbone with a
**Squeeze-and-Excitation (SE)** attention block, fine-tuned to distinguish
between {len(CLASSES)} lesion types: {", ".join(c.title() for c in CLASSES)}.

## Explainable AI

Predictions are accompanied by a **Grad-CAM++** heatmap, which highlights the
image regions that most influenced the model's decision. This is generated
via forward/backward hooks on the final convolutional block of the backbone,
producing a class-specific localization map that is overlaid on the original
image. Comparing the highlighted region to the actual lesion location helps
indicate whether the model is "looking at the right thing."

## Clinical RAG

The "Diagnostic Support Report" and "Clinical Assistant" are powered by a
**Retrieval-Augmented Generation (RAG)** pipeline:

1. A knowledge base of dermatology reference text (see "Knowledge Sources"
   below) is split into overlapping chunks.
2. Each chunk is embedded with the `all-MiniLM-L6-v2` sentence-transformer
   and indexed with **FAISS** for fast semantic search.
3. For a given prediction or question, the most relevant chunks are
   retrieved and used as grounding context for the report or answer.

## Local LLM Assistant

{"The Clinical Assistant uses the locally-loaded **" + LLM_NAME + "** model (via llama.cpp, CPU) to phrase answers in natural language, strictly grounded in the retrieved context." if _llm_model is not None else "No local generative LLM weights were found on this machine, so the Clinical Assistant currently runs in **retrieval-only (extractive) mode**: it returns the most relevant knowledge-base passages directly, with citations, instead of an LLM-generated paraphrase."}

Running entirely locally (rather than calling a cloud LLM API) means:

- **Privacy**: uploaded images and questions never leave this machine.
- **Offline capability**: the app works with no internet connection at
  inference time.
- **Reproducibility**: answers depend only on the bundled model weights and
  knowledge base, not on a third-party service that can change over time.

## Knowledge Sources

The knowledge base combines:

- **DermNet NZ** -- public dermatology reference articles, one per lesion
  class.
- **Curated summaries** (project-authored, based on DermNet NZ / AAD / WHO
  IARC) -- concise per-class descriptions.
- **Curated clinical warning signs** (project-authored) -- e.g. ABCDE
  criteria for melanoma.

Every retrieved passage used in a report or answer is cited with its source
name, URL, and access date.

## Safety Design

- **Hallucination prevention**: the Clinical Assistant is instructed to use
  *only* the retrieved context and to respond with "I could not find enough
  information in the knowledge base." if the context is insufficient.
- **Retrieval-first architecture**: if no local LLM is available, the system
  falls back to showing retrieved passages directly rather than guessing.
- **Citation requirements**: every report and chat answer lists its sources.
- **Confidence-based warnings**: predictions below 70% confidence show an
  uncertainty warning; melanoma predictions above 80% confidence show an
  urgent-care recommendation.
- **Language**: the system always shows "Predicted lesion type", never
  "Confirmed diagnosis".

## Limitations

- This system is **not a replacement for clinicians** -- it is a research
  and educational prototype.
- Predictions **may be incorrect**, including for malignant lesion types.
- **Image quality** (lighting, focus, framing, presence of hair/rulers) can
  significantly affect performance.
- Any finding from this tool should be followed by **clinical confirmation**
  from a qualified dermatologist.

## Ethical Considerations

- **Responsible AI usage**: this tool is intended to support learning about
  AI in healthcare, not to be used for self-diagnosis or to delay seeking
  medical care.
- **Transparency**: the model architecture, training data, evaluation
  metrics, and known limitations are documented in the project's Phase 8
  report (`skin_cancer_phase8.ipynb`).
- **Explainability**: Grad-CAM++ visualizations and retrieval citations are
  provided so users can see *why* a prediction or answer was given.
- **Human oversight**: all outputs are framed as decision-support
  information for a human (patient or clinician) to evaluate, not as
  autonomous decisions.

---

> {PERMANENT_DISCLAIMER}
"""


# ---------------------------------------------------------------------------
# Theme, dark mode, and screen-recording: CSS + client-side JS only
# (no Python state, no effect on inference/preprocessing/Grad-CAM/RAG)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ============================================================
   DERMASCAN AI — Professional Medical Interface
   ============================================================ */

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideInRight {
  from { opacity: 0; transform: translateX(16px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes rec-pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.35; }
}
@keyframes bar-grow {
  from { width: 0% !important; }
}
@keyframes shimmer {
  0%   { background-position: -400px 0; }
  100% { background-position:  400px 0; }
}

:root {
  --blue-900: #1e3a6e;
  --blue-700: #1d4ed8;
  --blue-600: #2563eb;
  --blue-500: #3b82f6;
  --blue-400: #60a5fa;
  --blue-100: #dbeafe;
  --blue-50 : #eff6ff;

  --teal-600: #0891b2;
  --teal-500: #06b6d4;
  --teal-100: #cffafe;

  --red-700 : #b91c1c;
  --red-600 : #dc2626;
  --red-100 : #fee2e2;
  --red-50  : #fff5f5;

  --amber-600: #d97706;
  --amber-100: #fef3c7;
  --amber-50 : #fffbeb;

  --green-700: #047857;
  --green-600: #059669;
  --green-100: #d1fae5;
  --green-50 : #f0fdf4;

  --gray-900: #0f172a;
  --gray-800: #1e293b;
  --gray-700: #334155;
  --gray-600: #475569;
  --gray-500: #64748b;
  --gray-400: #94a3b8;
  --gray-300: #cbd5e1;
  --gray-200: #e2e8f0;
  --gray-100: #f1f5f9;
  --gray-50 : #f8fafc;
  --white   : #ffffff;

  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  --radius-xl: 22px;

  --shadow-sm  : 0 1px 3px rgba(0,0,0,.07), 0 1px 2px rgba(0,0,0,.04);
  --shadow-md  : 0 4px 18px rgba(0,0,0,.09), 0 2px 6px rgba(0,0,0,.05);
  --shadow-lg  : 0 8px 36px rgba(0,0,0,.12), 0 4px 12px rgba(0,0,0,.07);
  --shadow-blue: 0 4px 22px rgba(59,130,246,.28);

  --transition: all 0.22s cubic-bezier(.4,0,.2,1);

  /* Semantic urgency */
  --urgent-bg    : var(--red-50);
  --urgent-border: var(--red-600);
  --urgent-text  : var(--red-700);
  --monitor-bg    : var(--amber-50);
  --monitor-border: var(--amber-600);
  --monitor-text  : var(--amber-600);
  --routine-bg    : var(--green-50);
  --routine-border: var(--green-600);
  --routine-text  : var(--green-700);
}

/* ── Base ─────────────────────────────────────────────────── */
.gradio-container {
  background: linear-gradient(150deg, #eef2fb 0%, #e8f1fe 40%, #ecf5fb 100%) !important;
  min-height: 100vh;
  font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}
html.dark .gradio-container,
.gradio-container.dark {
  background: linear-gradient(150deg, #060c18 0%, #0b1424 50%, #080e1c 100%) !important;
}

/* ── App header ──────────────────────────────────────────── */
.app-header {
  background: linear-gradient(135deg, #1a4fcf 0%, #0e7fb5 60%, #0a9ead 100%);
  border-radius: var(--radius-xl);
  padding: 22px 28px 18px;
  margin-bottom: 18px;
  box-shadow: var(--shadow-lg), 0 0 0 1px rgba(255,255,255,.12);
  animation: fadeInUp 0.45s ease both;
}
.app-header h1 {
  color: #ffffff !important;
  font-size: 1.55em !important;
  font-weight: 800 !important;
  letter-spacing: -.5px !important;
  margin: 0 0 4px !important;
  text-shadow: 0 1px 6px rgba(0,0,0,.18);
}
.app-header p, .app-header .subtitle {
  color: rgba(255,255,255,.84) !important;
  font-size: 0.88em !important;
  margin: 0 !important;
}

/* ── Toolbar ─────────────────────────────────────────────── */
.toolbar-row {
  background: var(--white);
  border-radius: var(--radius-md);
  border: 1px solid var(--gray-200);
  padding: 8px 14px;
  margin-bottom: 14px;
  box-shadow: var(--shadow-sm);
  display: flex;
  align-items: center;
  gap: 8px;
}
html.dark .toolbar-row {
  background: #111827;
  border-color: #1e2d45;
}

/* ── Buttons ─────────────────────────────────────────────── */
.gradio-container button {
  border-radius: var(--radius-md) !important;
  font-weight: 600 !important;
  letter-spacing: .15px;
  transition: var(--transition);
  font-size: 0.88em !important;
}
.gradio-container button:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md) !important;
}
.gradio-container button:active { transform: translateY(0) !important; }

/* Primary "Analyze" button */
.gradio-container button.primary,
button[variant="primary"] {
  background: linear-gradient(135deg, var(--blue-600) 0%, var(--teal-600) 100%) !important;
  border: none !important;
  color: #fff !important;
  box-shadow: var(--shadow-blue) !important;
  padding: 10px 26px !important;
  font-size: 0.96em !important;
  letter-spacing: .3px;
}
.gradio-container button.primary:hover,
button[variant="primary"]:hover {
  background: linear-gradient(135deg, #1a3fbf 0%, #0770a0 100%) !important;
  box-shadow: 0 6px 26px rgba(37,99,235,.40) !important;
}

/* Secondary / ghost buttons */
.gradio-container button:not(.primary) {
  background: var(--white) !important;
  border: 1.5px solid var(--gray-300) !important;
  color: var(--gray-700) !important;
}
html.dark .gradio-container button:not(.primary) {
  background: #1a2640 !important;
  border-color: #2e4160 !important;
  color: var(--gray-100) !important;
}

/* ── Tabs ────────────────────────────────────────────────── */
.gradio-container .tabs > .tab-nav {
  background: var(--white);
  border-radius: var(--radius-lg) var(--radius-lg) 0 0;
  border-bottom: 2px solid var(--gray-200);
  padding: 0 6px;
  box-shadow: var(--shadow-sm);
}
html.dark .gradio-container .tabs > .tab-nav {
  background: #111827;
  border-bottom-color: #1e2d45;
}
.gradio-container .tabs > .tab-nav button {
  background: transparent !important;
  border: none !important;
  border-bottom: 3px solid transparent !important;
  border-radius: 0 !important;
  color: var(--gray-500) !important;
  font-weight: 600 !important;
  padding: 12px 22px !important;
  transition: var(--transition);
  box-shadow: none !important;
  transform: none !important;
  font-size: 0.92em !important;
}
.gradio-container .tabs > .tab-nav button:hover {
  color: var(--blue-600) !important;
  transform: none !important;
  box-shadow: none !important;
}
.gradio-container .tabs > .tab-nav button.selected {
  color: var(--blue-600) !important;
  border-bottom: 3px solid var(--blue-600) !important;
  background: transparent !important;
}
html.dark .gradio-container .tabs > .tab-nav button       { color: var(--gray-400) !important; }
html.dark .gradio-container .tabs > .tab-nav button.selected {
  color: var(--blue-400) !important;
  border-bottom-color: var(--blue-400) !important;
}
.gradio-container .tabitem {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 16px 2px !important;
}

/* ── Section panels ──────────────────────────────────────── */
.gradio-container .form,
.gradio-container .block {
  border-radius: var(--radius-md) !important;
  transition: var(--transition);
}

/* Images */
.gradio-container .svelte-1ipelgc img {
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);
}

/* ── Report card grid ────────────────────────────────────── */
#clinical-report { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }

.report-header {
  margin-bottom: 16px;
  animation: fadeInUp 0.3s ease both;
}
.report-title {
  font-size: 1.18em;
  font-weight: 800;
  color: var(--gray-900);
  letter-spacing: -.3px;
}
.report-subtitle {
  font-size: 0.81em;
  color: var(--gray-500);
  margin-top: 2px;
}

.report-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
@media (max-width: 860px) { .report-grid { grid-template-columns: 1fr; } }

.report-card {
  background: var(--white);
  border: 1px solid var(--gray-200);
  border-left: 4px solid var(--blue-500);
  border-radius: var(--radius-md);
  padding: 12px 14px;
  box-shadow: var(--shadow-sm);
  animation: fadeInUp 0.35s ease both;
  transition: var(--transition);
}
.report-card:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
}
.report-card-wide { grid-column: 1 / -1; }

.report-card-title {
  font-size: 0.75em;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .8px;
  color: var(--gray-500);
  margin-bottom: 7px;
  display: flex;
  align-items: center;
  gap: 5px;
}
.report-card-body {
  color: var(--gray-800);
  font-size: 0.91em;
  line-height: 1.58;
}

/* Urgency variants */
.report-card--urgent  { border-left-color: var(--red-600);   background: var(--red-50);   }
.report-card--monitor { border-left-color: var(--amber-600); background: var(--amber-50); }
.report-card--routine { border-left-color: var(--green-600); background: var(--green-50); }
.report-card--positive{ border-left-color: var(--red-600);   background: #fff5f5; }
.report-card--negative{ border-left-color: var(--green-600); background: #f0fdf4; }

.report-card--urgent  .report-card-title { color: var(--red-700);   }
.report-card--urgent  .report-card-body  { color: var(--red-700);   }
.report-card--monitor .report-card-title { color: var(--amber-600); }
.report-card--monitor .report-card-body  { color: var(--amber-600); }
.report-card--routine .report-card-title { color: var(--green-700); }
.report-card--routine .report-card-body  { color: var(--green-700); }

/* Confidence bar */
.conf-pct {
  font-size: 1.65em;
  font-weight: 900;
  color: var(--blue-700);
  letter-spacing: -1.5px;
  line-height: 1.1;
}
.conf-bar-wrap {
  background: var(--gray-200);
  border-radius: 999px;
  height: 9px;
  overflow: hidden;
  margin: 8px 0 5px;
}
.conf-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--blue-600) 0%, var(--teal-500) 100%);
  border-radius: 999px;
  animation: bar-grow 0.9s cubic-bezier(.4,0,.2,1) both;
}

/* Prediction bars */
.pred-bar-item { margin-bottom: 10px; }
.pred-bar-label {
  display: flex;
  justify-content: space-between;
  font-size: 0.9em;
  font-weight: 600;
  color: var(--gray-800);
  margin-bottom: 4px;
}
.pred-bar-track {
  background: var(--gray-200);
  border-radius: 999px;
  height: 9px;
  overflow: hidden;
}
.pred-bar-fill {
  height: 100%;
  border-radius: 999px;
  animation: bar-grow 0.8s cubic-bezier(.4,0,.2,1) both;
}
.pred-rank-1 .pred-bar-fill { background: linear-gradient(90deg, var(--blue-600), var(--teal-500)); }
.pred-rank-2 .pred-bar-fill { background: linear-gradient(90deg, #4a7ecf, #62a8cf); opacity: .85; }
.pred-rank-3 .pred-bar-fill { background: linear-gradient(90deg, #7aa8cc, #9ecad8); opacity: .65; }

/* References / knowledge sources */
details.report-refs {
  background: var(--gray-50);
  border: 1px solid var(--gray-200);
  border-radius: var(--radius-md);
  padding: 10px 16px;
  margin-top: 12px;
  font-size: 0.84em;
}
details.report-refs summary {
  cursor: pointer;
  font-weight: 700;
  color: var(--blue-600);
  user-select: none;
  list-style: none;
}
details.report-refs summary::-webkit-details-marker { display: none; }
details.report-refs ul {
  margin: 8px 0 0;
  padding-left: 1.3em;
  color: var(--gray-600);
  line-height: 1.75;
}

/* Safety disclaimer */
.report-disclaimer {
  background: var(--gray-100);
  border: 1px solid var(--gray-300);
  border-radius: var(--radius-md);
  padding: 10px 16px;
  font-size: 0.81em;
  color: var(--gray-600);
  line-height: 1.55;
  margin-top: 12px;
}

/* ── Dark mode report ────────────────────────────────────── */
html.dark .report-header .report-title  { color: #e8f0fe; }
html.dark .report-header .report-subtitle { color: #7a9cc0; }

html.dark .report-card {
  background: #141e30;
  border-color: #1e3050;
  color: #c8d8ee;
}
html.dark .report-card-body  { color: #c0d2e8; }
html.dark .report-card-title { color: #7aa0c8; }

html.dark .report-card--urgent  { background: #250c0c; border-left-color: #ef4444; }
html.dark .report-card--monitor { background: #201600; border-left-color: #f59e0b; }
html.dark .report-card--routine { background: #092018; border-left-color: #10b981; }
html.dark .report-card--positive{ background: #200808; border-left-color: #ef4444; }
html.dark .report-card--negative{ background: #082018; border-left-color: #10b981; }

html.dark .report-card--urgent  .report-card-title,
html.dark .report-card--urgent  .report-card-body  { color: #fca5a5; }
html.dark .report-card--monitor .report-card-title,
html.dark .report-card--monitor .report-card-body  { color: #fcd34d; }
html.dark .report-card--routine .report-card-title,
html.dark .report-card--routine .report-card-body  { color: #6ee7b7; }

html.dark .conf-pct        { color: #60a5fa; }
html.dark .pred-bar-label  { color: #c0d2e8; }
html.dark .pred-bar-track,
html.dark .conf-bar-wrap   { background: #1e3050; }

html.dark .report-disclaimer {
  background: #141e30;
  border-color: #1e3050;
  color: #8aa0bc;
}
html.dark details.report-refs {
  background: #0e1828;
  border-color: #1e3050;
}
html.dark details.report-refs summary { color: #60a5fa; }
html.dark details.report-refs ul      { color: #8aa0bc; }

/* ── Permanent disclaimer banner ────────────────────────── */
#perm-disclaimer {
  background: linear-gradient(135deg, #fff9e6, #fff4d0);
  border: 1px solid #e8c54a;
  border-left: 4px solid var(--amber-600);
  border-radius: var(--radius-md);
  padding: 10px 16px;
  font-size: 0.84em;
  color: #7a5100;
  margin-bottom: 14px;
  animation: slideInRight 0.4s ease both;
}
html.dark #perm-disclaimer {
  background: #1a1200;
  border-color: #6b4400;
  color: #fcd34d;
}

/* ── Chatbot ─────────────────────────────────────────────── */
.gradio-container .chatbot {
  border-radius: var(--radius-lg) !important;
  border: 1px solid var(--gray-200) !important;
  box-shadow: var(--shadow-md) !important;
}
html.dark .gradio-container .chatbot {
  background: #0e1828 !important;
  border-color: #1e3050 !important;
}

/* ── Recording status ────────────────────────────────────── */
#recording-status {
  font-size: 0.82em;
  font-weight: 700;
  padding: 3px 12px;
  border-radius: 999px;
  background: var(--gray-100);
  color: var(--gray-600);
  display: inline-block;
  transition: var(--transition);
}
#recording-status[data-active="true"] {
  color: var(--red-600);
  background: var(--red-50);
  animation: rec-pulse 1.2s ease infinite;
}

/* ── Input elements ──────────────────────────────────────── */
.gradio-container label span {
  font-weight: 600 !important;
  font-size: 0.87em !important;
  color: var(--gray-700) !important;
}
html.dark .gradio-container label span { color: var(--gray-400) !important; }
.gradio-container textarea, .gradio-container input[type="text"] {
  border-radius: var(--radius-md) !important;
  border: 1.5px solid var(--gray-300) !important;
  transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.gradio-container textarea:focus, .gradio-container input[type="text"]:focus {
  border-color: var(--blue-500) !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,.15) !important;
}
html.dark .gradio-container textarea,
html.dark .gradio-container input[type="text"] {
  border-color: #2e4160 !important;
  background: #0e1828 !important;
  color: #c0d2e8 !important;
}
"""

# Client-side helpers for the screen-recording buttons. Pure browser
# MediaRecorder + getDisplayMedia -- no server round-trip, no new
# dependencies, no ffmpeg/transcoding.
HEAD_SCRIPT = """
<script>
window.__screenRecorder = null;
window.__screenChunks = [];

window.__startScreenRecording = async function() {
    try {
        const stream = await navigator.mediaDevices.getDisplayMedia({video: true, audio: true});
        window.__screenChunks = [];
        const recorder = new MediaRecorder(stream);
        recorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) window.__screenChunks.push(e.data);
        };
        recorder.onstop = () => {
            const blob = new Blob(window.__screenChunks, {type: "video/webm"});
            const url = URL.createObjectURL(blob);
            const now = new Date();
            const pad = (n) => String(n).padStart(2, "0");
            const name = "screen_record_" + now.getFullYear() + "_" + pad(now.getMonth() + 1) + "_" +
                pad(now.getDate()) + "_" + pad(now.getHours()) + "_" + pad(now.getMinutes()) + ".webm";
            const a = document.createElement("a");
            a.href = url;
            a.download = name;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            stream.getTracks().forEach((t) => t.stop());
            const status = document.getElementById("recording-status");
            if (status) { status.textContent = "● Idle"; status.removeAttribute("data-active"); }
        };
        stream.getTracks()[0].onended = () => {
            if (window.__screenRecorder && window.__screenRecorder.state !== "inactive")
                window.__screenRecorder.stop();
        };
        window.__screenRecorder = recorder;
        // pick best supported mime type
        const mimeType = ["video/webm;codecs=vp9","video/webm;codecs=vp8","video/webm"]
            .find(t => MediaRecorder.isTypeSupported(t)) || "";
        recorder.start();
        const status = document.getElementById("recording-status");
        if (status) { status.textContent = "⏺ Recording"; status.setAttribute("data-active","true"); }
    } catch (err) {
        console.error("Screen recording failed to start", err);
        const status = document.getElementById("recording-status");
        if (status) status.textContent = "Permission denied";
    }
};

window.__stopScreenRecording = function() {
    if (window.__screenRecorder && window.__screenRecorder.state !== "inactive") {
        window.__screenRecorder.stop();
    }
};

window.__toggleDarkMode = function() {
    document.documentElement.classList.toggle("dark");
    const container = document.querySelector(".gradio-container");
    if (container) container.classList.toggle("dark");
};
</script>
"""


def build_ui():
    with gr.Blocks(title="DermaScan AI — Skin Lesion Analysis") as demo:

        # ── App header ────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
          <h1>🔬 DermaScan AI</h1>
          <p class="subtitle">
            EfficientNet-B4 · Grad-CAM++ · Clinical RAG · Explainable AI &nbsp;—&nbsp; Research &amp; Education Demo
          </p>
        </div>
        """)

        # ── Toolbar ───────────────────────────────────────────────────────
        with gr.Row():
            dark_toggle_btn = gr.Button("🌙 Dark Mode", size="sm")
            start_rec_btn   = gr.Button("⏺ Record",     size="sm")
            stop_rec_btn    = gr.Button("⏹ Stop",       size="sm")
            gr.HTML('<span id="recording-status">● Idle</span>')

        dark_toggle_btn.click(fn=None, inputs=None, outputs=None,
                              js="() => window.__toggleDarkMode()")
        start_rec_btn.click(fn=None, inputs=None, outputs=None,
                            js="() => window.__startScreenRecording()")
        stop_rec_btn.click(fn=None, inputs=None, outputs=None,
                           js="() => window.__stopScreenRecording()")

        # ── Disclaimer banner ─────────────────────────────────────────────
        gr.HTML(f'<div id="perm-disclaimer">⚕ {PERMANENT_DISCLAIMER}</div>')

        llm_status = (
            f"Clinical Assistant LLM: **{LLM_NAME}** (local, CPU via llama.cpp)"
            if _llm_model is not None
            else "_Clinical Assistant: retrieval-only mode (no local LLM weights found) — "
                 "answers are shown as grounded knowledge-base passages._"
        )
        gr.Markdown(llm_status)

        diagnosis_state = gr.State({})

        with gr.Tabs():
            # ── Tab 1: Diagnosis ──────────────────────────────────────────
            with gr.Tab("🔬 Diagnosis"):

                with gr.Row(equal_height=False):
                    # Left column — upload + images
                    with gr.Column(scale=1, min_width=260):
                        image_in = gr.Image(
                            type="pil",
                            label="Upload Skin Lesion Image",
                            sources=["upload", "webcam", "clipboard"],
                        )
                        analyze_btn = gr.Button("🔍 Analyze", variant="primary")

                        gr.Markdown("#### Visual Analysis")
                        original_out     = gr.Image(label="Preprocessed Image",    interactive=False)
                        gradcam_out      = gr.Image(label="Grad-CAM++ Heatmap",    interactive=False)
                        side_by_side_out = gr.Image(label="Side-by-Side Overlay",  interactive=False)

                    # Middle column — predictions
                    with gr.Column(scale=1, min_width=220):
                        gr.Markdown("#### Top-3 Predictions")
                        top3_html = gr.HTML(
                            value="<div style='color:#94a3b8;font-size:0.9em;padding:8px 0;'>"
                                  "Predictions will appear here after analysis.</div>"
                        )

                    # Right column — report + follow-up
                    with gr.Column(scale=2, min_width=340):
                        report_html = gr.HTML(
                            value=_REPORT_PLACEHOLDER,
                            elem_id="clinical-report",
                        )
                        with gr.Group(visible=False) as followup_group:
                            gr.Markdown("#### Ask About This Diagnosis")
                            gr.Markdown(
                                '_Examples: "What is this condition?", '
                                '"Is it dangerous?", "What should I do next?"_'
                            )
                            followup_q = gr.Textbox(
                                label="Your question",
                                placeholder="Ask a question about this result...",
                                lines=1,
                            )
                            followup_btn = gr.Button("Ask", variant="primary")
                            followup_a   = gr.Markdown()

                analyze_btn.click(
                    fn=run_diagnosis,
                    inputs=image_in,
                    outputs=[
                        original_out, gradcam_out, side_by_side_out,
                        top3_html, report_html, followup_group, diagnosis_state,
                    ],
                )
                followup_btn.click(
                    fn=run_followup,
                    inputs=[followup_q, diagnosis_state],
                    outputs=followup_a,
                )
                followup_q.submit(
                    fn=run_followup,
                    inputs=[followup_q, diagnosis_state],
                    outputs=followup_a,
                )

            # ── Tab 2: Clinical Assistant ─────────────────────────────────
            with gr.Tab("💬 Clinical Assistant"):
                gr.Markdown(
                    "Ask any question about skin lesions — explanations, risk factors, "
                    "symptoms, prevention, follow-up care, or how the model works. "
                    "Answers are grounded only in the local clinical knowledge base."
                )
                chatbot = gr.Chatbot(
                    height=460,
                    label="Clinical Assistant",
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="",
                        placeholder="e.g. What are the warning signs of melanoma?",
                        scale=5,
                        lines=1,
                    )
                    chat_submit = gr.Button("Send", variant="primary", scale=1)
                chat_clear = gr.Button("Clear conversation", size="sm")

                chat_submit.click(
                    fn=run_chat,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input],
                )
                chat_input.submit(
                    fn=run_chat,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input],
                )
                chat_clear.click(fn=lambda: ([], ""), outputs=[chatbot, chat_input])

            # ── Tab 3: About & Safety ─────────────────────────────────────
            with gr.Tab("📖 About & Safety"):
                gr.Markdown(ABOUT_TEXT)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    share = os.environ.get("GRADIO_SHARE", "false").strip().lower() == "true"
    demo.launch(share=share, theme=gr.themes.Soft(), css=CUSTOM_CSS, head=HEAD_SCRIPT)
