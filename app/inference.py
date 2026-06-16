"""Self-contained inference + clinical RAG module for the Gradio demo.

Loads the trained EfficientNet-B4 + SE-block classifier, the Grad-CAM++
implementation, and the pre-built FAISS knowledge base from Phase 7, and
exposes a single entry point: predict_and_explain(image).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import cv2
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_b4

APP_DIR = Path(__file__).resolve().parent

SEED = 42
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(APP_DIR / 'config' / 'preprocessing_config.json') as f:
    _cfg = json.load(f)

CLASSES = _cfg['classes']
NUM_CLASSES = _cfg['num_classes']
NORM_MEAN = _cfg['norm_mean']
NORM_STD = _cfg['norm_std']
IMG_SIZE = _cfg['img_size']
MALIGNANT_CLASSES = ['melanoma', 'basal cell carcinoma', 'squamous cell carcinoma']


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SEBlock(nn.Module):
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


_model = SkinCancerModel(num_classes=NUM_CLASSES).to(DEVICE)
_state_dict = torch.load(APP_DIR / 'model' / 'final_model.pth', map_location=DEVICE, weights_only=True)
_model.load_state_dict(_state_dict)
_model.eval()


# ---------------------------------------------------------------------------
# Grad-CAM++
# ---------------------------------------------------------------------------
class GradCAMPlusPlus:
    """Grad-CAM++ via forward/backward hooks on a target conv layer."""

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
        cam_t = F.interpolate(cam_t, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam_t.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)
        return cam, output.softmax(dim=1).detach().cpu().numpy()[0]


_gradcam = GradCAMPlusPlus(_model, _model.features[-1])

inference_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


def load_image_tensor(image: Image.Image):
    image = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    tensor = inference_transform(image).unsqueeze(0).to(DEVICE)
    return image, tensor


def overlay_heatmap(image_pil, cam, alpha=0.45):
    image_np = np.asarray(image_pil).astype(np.float32) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = alpha * heatmap + (1 - alpha) * image_np
    return np.clip(overlay, 0, 1)


def describe_gradcam(cam: np.ndarray, threshold: float = 0.5) -> str:
    """Turn a Grad-CAM++ heatmap into a short natural-language description."""
    h, w = cam.shape
    mask = cam >= threshold
    if not mask.any():
        return ("The model's attention map (Grad-CAM++) was diffuse, with no single "
                "dominant focus region. Compare the heatmap overlay to the lesion "
                "location to assess whether the model attended to the lesion.")

    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean() / h, xs.mean() / w
    coverage = mask.mean()

    vert = 'upper' if cy < 0.4 else ('lower' if cy > 0.6 else 'central')
    horiz = 'left' if cx < 0.4 else ('right' if cx > 0.6 else 'central')
    if vert == 'central' and horiz == 'central':
        position = 'the central region'
    elif vert == 'central':
        position = f'the {horiz} side'
    elif horiz == 'central':
        position = f'the {vert} part'
    else:
        position = f'the {vert}-{horiz} region'

    extent = 'a tightly focused area' if coverage < 0.20 else (
        'a moderate area' if coverage < 0.45 else 'a broad area')

    return (f"The model's attention (Grad-CAM++) was concentrated on {extent} in "
            f"{position} of the image, covering approximately {coverage * 100:.0f}% "
            f"of the frame at high activation. Compare this overlay to the lesion's "
            f"location in the original image -- attention concentrated on the lesion "
            f"itself supports the prediction, while attention on skin, hair, or "
            f"borders would indicate the prediction may be unreliable.")


# ---------------------------------------------------------------------------
# Clinical RAG (Phase 7 knowledge base, pre-built)
# ---------------------------------------------------------------------------
_embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
_faiss_index = faiss.read_index(str(APP_DIR / 'rag' / 'faiss_index.bin'))
with open(APP_DIR / 'rag' / 'chunk_metadata.json', encoding='utf-8') as f:
    _chunk_records = json.load(f)


def semantic_search(query: str, k: int = 5):
    q_emb = _embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    distances, indices = _faiss_index.search(q_emb, k)
    return [(_chunk_records[i], float(d)) for i, d in zip(indices[0], distances[0])]


@dataclass
class RetrievedContext:
    predicted_class: str
    confidence: float
    chunks: list = field(default_factory=list)
    sources: list = field(default_factory=list)


def retrieve_for_class(predicted_class: str, confidence: float, k: int = 3) -> RetrievedContext:
    class_chunks = [c for c in _chunk_records if c['class'] == predicted_class]
    semantic_chunks = [c for c, _ in semantic_search(predicted_class, k=k)]

    seen_ids = set()
    combined = []
    for c in class_chunks + semantic_chunks:
        if c['chunk_id'] not in seen_ids:
            seen_ids.add(c['chunk_id'])
            combined.append(c)

    seen_src = set()
    sources = []
    for c in combined:
        key = (c['source'], c['source_url'])
        if key not in seen_src:
            seen_src.add(key)
            sources.append(f"{c['source']} -- {c['source_url']} (accessed {c['source_date']})")

    return RetrievedContext(predicted_class=predicted_class, confidence=confidence,
                             chunks=combined, sources=sources)


URGENCY_LABELS = {
    'URGENT': 'URGENT - prompt dermatologist evaluation recommended',
    'MONITOR': 'MONITOR - dermatologist review recommended (not an emergency)',
    'ROUTINE': 'ROUTINE - routine self-monitoring is sufficient',
}

URGENCY_ACTIONS = {
    'URGENT': (
        "This finding is associated with a malignant or potentially serious "
        "lesion. Prompt evaluation (within days) by a dermatologist is "
        "recommended, including biopsy if indicated."
    ),
    'MONITOR': (
        "This finding may represent a precancerous lesion. A dermatologist "
        "should evaluate it; early treatment can prevent progression to "
        "skin cancer."
    ),
    'ROUTINE': (
        "This finding is typically benign. Routine self-monitoring is "
        "sufficient. Consult a dermatologist if the lesion changes in size, "
        "shape, color, or becomes symptomatic."
    ),
}

DISCLAIMER = (
    "DISCLAIMER: This report was generated automatically by an AI research "
    "prototype for an educational graduation project. It is NOT a medical "
    "diagnosis and has NOT been validated for clinical use. All findings must "
    "be reviewed by a qualified dermatologist or physician. Do not delay "
    "seeking medical care based on this output."
)


def generate_report(predicted_class: str, confidence: float, top3: list,
                     gradcam_description: str, context: RetrievedContext) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("DIAGNOSTIC SUPPORT REPORT")
    lines.append("=" * 70)

    lines.append(f"\nPredicted Condition: {predicted_class.title()}")
    lines.append(f"Confidence: {confidence * 100:.1f}%")

    class_chunks = [c for c in context.chunks if c['class'] == predicted_class]
    urgency = class_chunks[0]['urgency'] if class_chunks else 'ROUTINE'
    lines.append(f"Urgency Level: {URGENCY_LABELS.get(urgency, urgency)}")

    lines.append("\n--- What Is This? ---")
    desc_chunk = next((c for c in class_chunks if c['source'].startswith('Curated summary')), None)
    if desc_chunk is None:
        desc_chunk = class_chunks[0] if class_chunks else None
    if desc_chunk:
        lines.append(desc_chunk['text'])
    else:
        lines.append(
            "No knowledge-base content is available for this class (offline "
            "fallback). Please consult a dermatologist for information about "
            "this condition."
        )

    lines.append("\n--- Visual Characteristics (Grad-CAM++) ---")
    lines.append(gradcam_description)

    lines.append("\n--- Clinical Recommendation ---")
    lines.append(URGENCY_ACTIONS.get(urgency, URGENCY_ACTIONS['ROUTINE']))

    lines.append("\n--- Top Differential Diagnoses ---")
    for cls, p in top3:
        lines.append(f"  - {cls.title()}: {p * 100:.1f}%")

    lines.append("\n--- Important Sources ---")
    if context.sources:
        for s in context.sources:
            lines.append(f"  - {s}")
    else:
        lines.append("  - No sources retrieved (offline fallback).")

    if confidence < 0.70:
        lines.append(
            "\n[NOTE] Model confidence is below 70%. This prediction is "
            "uncertain and should not be relied upon -- please consult a "
            "dermatologist for an accurate diagnosis."
        )

    if predicted_class == 'melanoma' and confidence > 0.80:
        lines.append(
            "\n[WARNING] High-confidence melanoma prediction. SEEK IMMEDIATE "
            "MEDICAL CARE from a dermatologist or healthcare provider for "
            "evaluation and biopsy."
        )

    lines.append("\n" + "=" * 70)
    lines.append(DISCLAIMER)
    lines.append("=" * 70)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# End-to-end entry point
# ---------------------------------------------------------------------------
def predict_and_explain(image: Image.Image) -> dict:
    image_pil, tensor = load_image_tensor(image)

    with torch.no_grad():
        logits = _model(tensor)
        proba = logits.softmax(dim=1).cpu().numpy()[0]

    top3_idx = np.argsort(-proba)[:3]
    top3 = [(CLASSES[i], float(proba[i])) for i in top3_idx]
    pred_idx = int(top3_idx[0])
    pred_class = CLASSES[pred_idx]
    confidence = float(proba[pred_idx])

    cam, _ = _gradcam.generate(tensor, pred_idx)
    overlay = overlay_heatmap(image_pil, cam)
    gradcam_description = describe_gradcam(cam)

    context = retrieve_for_class(pred_class, confidence)
    report = generate_report(pred_class, confidence, top3, gradcam_description, context)

    return {
        'prediction': pred_class,
        'confidence': confidence,
        'top3': top3,
        'gradcam_image': overlay,
        'medical_report': report,
        'original_image': image_pil,
    }
