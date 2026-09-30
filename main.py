"""
main.py — MMBCNet FastAPI Backend
==================================
Endpoints:
  POST /predict/mammo          → single image → pred + conf + flag + gradcam (base64)
  POST /predict/sono           → single image → pred + conf + flag + gradcam (base64)
  POST /predict/mammo/bulk     → multiple images → JSON list (pred, conf, flag)
  POST /predict/sono/bulk      → multiple images → JSON list (pred, conf, flag)
"""

import os, io, base64, warnings, tempfile
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import timm
import torchmetrics
import lightning as L
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.filters import threshold_otsu
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# DICOM PARSER
# ══════════════════════════════════════════════════════════════════════════════

def _str(tag):
    """Safely convert a DICOM tag value to a stripped string."""
    try:
        return str(tag).strip()
    except Exception:
        return ""

def parse_dicom(raw: bytes):
    """
    Parse a DICOM byte stream.
    Returns:
      image_bytes : JPEG bytes suitable for model inference
      metadata    : dict of patient / study fields
    """
    if not PYDICOM_AVAILABLE:
        raise RuntimeError("pydicom is not installed.")
    return _dicom_to_jpeg(raw)


def _is_dicom(raw: bytes, filename: str) -> bool:
    """Heuristic: check DICOM magic bytes or .dcm extension."""
    is_dcm_ext = filename.lower().endswith(".dcm")
    is_dcm_magic = len(raw) > 132 and raw[128:132] == b"DICM"
    return is_dcm_ext or is_dcm_magic


def try_parse_as_dicom(raw: bytes):
    """
    Attempt to parse raw bytes as DICOM (even without the DICM preamble).
    Returns (image_bytes, metadata) on success, or raises InvalidDicomError.
    Used as fallback for files that aren't recognized standard images.
    """
    if not PYDICOM_AVAILABLE:
        raise RuntimeError("pydicom is not installed.")
    # force=True allows parsing old ACR-NEMA DICOMs without the 128-byte preamble
    ds = pydicom.dcmread(io.BytesIO(raw), force=True)
    # Confirm it actually has pixel data — otherwise it's not a valid image DICOM
    if not hasattr(ds, 'PixelData'):
        raise ValueError("No pixel data in DICOM file.")
    return parse_dicom.__wrapped__(raw, ds) if hasattr(parse_dicom, '__wrapped__') else _dicom_to_jpeg(raw)


def _dicom_to_jpeg(raw: bytes, ds=None):
    """Convert DICOM pixel data to JPEG + extract metadata. Accepts pre-parsed ds."""
    if ds is None:
        ds = pydicom.dcmread(io.BytesIO(raw), force=True)

    def tag(name, default=""):
        val = getattr(ds, name, None)
        return _str(val) if val is not None else default

    metadata = {
        "patient_name"       : tag("PatientName"),
        "patient_id"         : tag("PatientID"),
        "patient_dob"        : tag("PatientBirthDate"),
        "patient_sex"        : tag("PatientSex"),
        "study_date"         : tag("StudyDate"),
        "study_description"  : tag("StudyDescription"),
        "modality"           : tag("Modality"),
        "accession_number"   : tag("AccessionNumber"),
        "institution"        : tag("InstitutionName"),
        "referring_physician": tag("ReferringPhysicianName"),
    }

    pixel_array = ds.pixel_array
    try:
        pixel_array = apply_voi_lut(pixel_array, ds)
    except Exception:
        pass

    pmin, pmax = float(pixel_array.min()), float(pixel_array.max())
    if pmax > pmin:
        pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
    else:
        pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)

    if pixel_array.ndim == 2:
        pixel_array = np.stack([pixel_array] * 3, axis=-1)
    elif pixel_array.shape[-1] == 1:
        pixel_array = np.concatenate([pixel_array] * 3, axis=-1)

    buf = io.BytesIO()
    Image.fromarray(pixel_array).save(buf, format="JPEG", quality=95)
    return buf.getvalue(), metadata


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MAMMO_CKPT_PATH = os.getenv("MAMMO_CKPT_PATH", str(BASE_DIR / "phase2-epoch=09-val_auc=0.9036.ckpt"))
SONO_MODEL_PATH  = os.getenv("SONO_MODEL_PATH",  str(BASE_DIR / "Frozen92.16.keras"))

# ── Constants ─────────────────────────────────────────────────────────────────
MAMMO_CLASSES  = ["Normal", "Benign", "Suspicious Malignant", "Malignant"]
SONO_CLASSES   = ["Benign", "Malignant", "Normal"]
MAMMO_IMG_SIZE = 512
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_MEAN   = 26.0
TARGET_STD    = 49.0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

MAMMO_IDX_NORMAL  = 0
MAMMO_IDX_BENIGN  = 1
MAMMO_IDX_SUSPMAL = 2
MAMMO_IDX_MAL     = 3
MAMMO_MAL_CONF_THRESH  = 0.50
MAMMO_SPECTRUM_THRESH  = 0.35
SONO_THRESH_DEFAULT    = 0.70
SONO_THRESH_MALIGNANT  = 0.60

# ── Global model holders ──────────────────────────────────────────────────────
mammo_model         = None
mammo_target_layers = None
mammo_transform     = None
sono_model          = None
sono_last_conv_layer= None
sono_gradcam_model  = None   # sub-model: (inputs) → [target_layer_output, predictions]
SONO_IMG_SIZE       = 300

# ── Sono Grad-CAM++ heatmap quality knobs (match notebook defaults) ────────────
SONO_HEATMAP_THRESHOLD_PERCENTILE = 45   # suppress activations below this percentile
SONO_HEATMAP_GAMMA                = 1.5  # >1 sharpens the peak, fades the tail faster
SONO_HEATMAP_BLUR_KERNEL          = 15   # Gaussian blur kernel size (odd int, 0 = off)
SONO_OVERLAY_MAX_ALPHA            = 0.65 # blend strength at peak activation


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

class ClinicalConvNeXt(L.LightningModule):
    def __init__(self, num_classes=4, lr=1e-5, class_weights=None, freeze_backbone=False):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.lr          = lr
        self.num_classes = num_classes
        self.backbone    = timm.create_model("convnext_small", pretrained=False, num_classes=0)
        backbone_features = self.backbone.num_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        for split in ["val", "test"]:
            for metric in ["acc", "auc", "f1", "recall", "specificity"]:
                setattr(self, f"{split}_{metric}",
                    getattr(torchmetrics,
                        {"acc": "Accuracy", "auc": "AUROC", "f1": "F1Score",
                         "recall": "Recall", "specificity": "Specificity"}[metric]
                    )(task="multiclass", num_classes=num_classes, average="macro"))

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mammo_model, mammo_target_layers, mammo_transform
    global sono_model, sono_last_conv_layer, SONO_IMG_SIZE

    # ── Mammography ───────────────────────────────────────────
    if not os.path.exists(MAMMO_CKPT_PATH):
        raise RuntimeError(f"Mammo checkpoint not found: {MAMMO_CKPT_PATH}")
    mammo_model = ClinicalConvNeXt.load_from_checkpoint(
        MAMMO_CKPT_PATH,
        num_classes=len(MAMMO_CLASSES),
        class_weights=None,
        map_location="cpu",
        strict=False,
    )
    mammo_model = mammo_model.to(DEVICE)
    mammo_model.eval()
    mammo_target_layers = [mammo_model.backbone.stages[-1].blocks[-1]]
    mammo_transform = A.Compose([
        A.Resize(MAMMO_IMG_SIZE, MAMMO_IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    print(f"[startup] Mammo model loaded on {DEVICE}")

    # ── Sonography ────────────────────────────────────────────
    if not os.path.exists(SONO_MODEL_PATH):
        raise RuntimeError(f"Sono model not found: {SONO_MODEL_PATH}")
    
    sono_model = tf.keras.models.load_model(
        SONO_MODEL_PATH,
        compile=False,
        custom_objects={'Custom>mish': tf.keras.activations.mish, 'mish': tf.keras.activations.mish}
    )
    sono_model.trainable = True
    if sono_model.input_shape[1] is not None:
        SONO_IMG_SIZE = sono_model.input_shape[1]

    # ── Auto-detect the best Grad-CAM++ target layer (notebook heuristic) ─────
    sono_last_conv_layer = _sono_find_gradcam_target_layer(sono_model)
    assert sono_last_conv_layer, "No spatial feature layer found in sono model."

    # Build the sub-model used by sono_gradcam_pp: outputs are
    # (target_conv_layer_output, final_predictions) – identical to the notebook.
    global sono_gradcam_model
    sono_gradcam_model = tf.keras.models.Model(
        inputs=sono_model.inputs,
        outputs=[sono_model.get_layer(sono_last_conv_layer).output, sono_model.output],
    )
    print(f"[startup] Sono model loaded  | input={SONO_IMG_SIZE}px | gradcam++={sono_last_conv_layer}")

    yield

    print("[shutdown] Models released.")


app = FastAPI(
    title="MMBCNet Inference API",
    description="Breast cancer detection — mammography & sonography",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def apply_clahe(image_uint8, clip_limit=1.5, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to a single RGB image.
    """
    lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge(
        [l_enhanced, a_channel, b_channel]
    )
    result = cv2.cvtColor(
        lab_enhanced,
        cv2.COLOR_LAB2RGB
    )

    return result

def tta_predict(model, images):
    """
    Test Time Augmentation (TTA) - Exact match to user colab
    """
    preds = []
    preds.append(model.predict(images, verbose=0))
    preds.append(model.predict(tf.image.flip_left_right(images), verbose=0))
    preds.append(model.predict(tf.image.flip_up_down(images), verbose=0))
    return np.mean(preds, axis=0)

def preprocess_mammogram_image(source, target_size: int = MAMMO_IMG_SIZE) -> np.ndarray:
    """JPG/JPEG → (512,512,3) uint8"""
    if isinstance(source, bytes):
        img = Image.open(io.BytesIO(source)).convert("L")
    elif isinstance(source, Image.Image):
        img = source.convert("L")
    else:
        raise TypeError(f"Expected bytes or PIL Image, got {type(source)}")

    img_array = np.array(img).astype(np.float32)
    h, w = img_array.shape

    # Nullify corner watermark
    top_rows   = int(h * 0.04)
    right_cols = int(w * 0.15)
    corner     = img_array[:top_rows, w - right_cols:]
    if corner.mean() > 1.5 * img_array.mean():
        img_array[:top_rows, w - right_cols:] = 0.0

    # Otsu breast crop
    try:
        thresh     = threshold_otsu(img_array)
        mask       = img_array > thresh
        rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
        r0, r1     = np.where(rows)[0][[0, -1]]
        c0, c1     = np.where(cols)[0][[0, -1]]
        mh = int((r1 - r0) * 0.05);  mw = int((c1 - c0) * 0.05)
        img_array  = img_array[max(0,r0-mh):min(h-1,r1+mh)+1,
                               max(0,c0-mw):min(w-1,c1+mw)+1]
    except Exception:
        pass

    # Normalize → brightness match
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255.0
    img_array = (img_array - img_array.mean()) / (img_array.std() + 1e-8) * TARGET_STD + TARGET_MEAN
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    # Aspect-ratio preserving resize + zero-pad
    h, w    = img_array.shape
    scale   = target_size / max(h, w)
    nh, nw  = int(h * scale), int(w * scale)
    img_pil = Image.fromarray(img_array).resize((nw, nh), Image.BICUBIC)
    padded  = np.zeros((target_size, target_size), dtype=np.uint8)
    ph, pw  = (target_size - nh) // 2, (target_size - nw) // 2
    padded[ph:ph+nh, pw:pw+nw] = np.array(img_pil)

    return np.stack([padded] * 3, axis=-1)   # (512,512,3) uint8


def preprocess_sono_image(raw: bytes):
    """
    Exact preprocessing from user's Colab training notebook
    """
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    
    img = img.resize((SONO_IMG_SIZE, SONO_IMG_SIZE))
    img = np.array(img).astype(np.float32)

    # Apply CLAHE if used during training
    img = apply_clahe(img.astype(np.uint8)).astype(np.float32)

    return img   # (H, W, 3) float32 in [0, 255]


# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM++
# ══════════════════════════════════════════════════════════════════════════════

def mammo_denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return torch.clamp(tensor.cpu() * std + mean, 0, 1)


def mammo_gradcam_pp(image_tensor):
    cam = GradCAMPlusPlus(model=mammo_model, target_layers=mammo_target_layers)
    mammo_model.eval()
    with torch.no_grad():
        logits     = mammo_model(image_tensor.to(DEVICE))
        probs      = torch.softmax(logits, dim=1)
        pred_class = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, pred_class].item())
    with torch.enable_grad():
        grayscale_cam = cam(
            input_tensor=image_tensor.to(DEVICE),
            targets=[ClassifierOutputTarget(pred_class)]
        )[0]
    rgb_img   = mammo_denormalize(image_tensor[0]).permute(1, 2, 0).numpy()
    cam_image = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True)
    return cam_image, pred_class, confidence, grayscale_cam, probs.cpu().numpy()[0]


# ══════════════════════════════════════════════════════════════════════════════
# SONO GRAD-CAM++ HELPERS  (ported from GradCAMPP_MMBCNet_Sonography notebook)
# ══════════════════════════════════════════════════════════════════════════════

def _sono_layer_output_shape(layer):
    """Return the output shape tuple of a Keras layer, or None on failure."""
    try:
        shape = layer.output_shape
    except Exception:
        try:
            shape = layer.output.shape
        except Exception:
            return None
    if isinstance(shape, list):
        shape = shape[0]
    return tuple(shape)


def _sono_find_gradcam_target_layer(model):
    """
    Auto-detect the best Grad-CAM++ target layer using the same heuristic as
    the notebook: the last 4-D feature-map layer that sits before the final
    GAP/GMP pooling stage (the one feeding the classifier head).
    """
    try:
        from tensorflow.keras import layers as tf_layers
        pooling_types = (
            tf_layers.GlobalAveragePooling2D,
            tf_layers.GlobalMaxPooling2D,
        )
    except Exception:
        pooling_types = ()

    pooling_indices = [
        i for i, l in enumerate(model.layers)
        if pooling_types and isinstance(l, pooling_types)
    ]

    four_d_candidates = [
        (i, l.name, _sono_layer_output_shape(l))
        for i, l in enumerate(model.layers)
        if _sono_layer_output_shape(l) is not None
        and len(_sono_layer_output_shape(l)) == 4
    ]

    if not four_d_candidates:
        # Absolute fallback: first spatial layer with output > 1px on each dim
        for layer in reversed(model.layers):
            try:
                s = layer.output.shape
                if len(s) == 4 and s[1] is not None and s[1] > 1:
                    return layer.name
            except Exception:
                continue
        return None

    if pooling_indices:
        # Anchor on the LAST pair of pooling layers (avoids CBAM's internal GAP)
        cutoff = (
            min(pooling_indices[-2:])
            if len(pooling_indices) >= 2
            else pooling_indices[-1]
        )
        before_pool = [c for c in four_d_candidates if c[0] < cutoff]
        if before_pool:
            chosen = max(before_pool, key=lambda c: c[0])
            return chosen[1]

    # Fallback: last 4-D layer in the whole model
    return four_d_candidates[-1][1]


def _sono_overlay_heatmap(
    original_uint8, heatmap,
    colormap=cv2.COLORMAP_JET,
    threshold_percentile=SONO_HEATMAP_THRESHOLD_PERCENTILE,
    gamma=SONO_HEATMAP_GAMMA,
    blur_ksize=SONO_HEATMAP_BLUR_KERNEL,
    max_alpha=SONO_OVERLAY_MAX_ALPHA,
):
    """
    Blend a normalised heatmap (values 0-1) onto an RGB uint8 image.
    Matches the overlay_heatmap() function from the notebook exactly.
    """
    h = np.clip(heatmap, 0.0, 1.0).astype(np.float32)

    # Suppress low-activation background noise
    if threshold_percentile and threshold_percentile > 0:
        thresh_val = np.percentile(h, threshold_percentile)
        h = np.where(h >= thresh_val, h - thresh_val, 0.0)
        max_val = h.max()
        if max_val > 1e-8:
            h = h / max_val

    # Smooth surviving blobs
    if blur_ksize and blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        h = cv2.GaussianBlur(h, (k, k), 0)
        h = np.clip(h, 0.0, 1.0)

    # Sharpen contrast
    if gamma and gamma != 1.0:
        h = np.power(h, gamma)

    heatmap_uint8 = np.uint8(255 * h)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Per-pixel alpha: activated regions are strongly coloured, background stays
    alpha_map = (h * max_alpha)[..., None]
    overlay = (
        original_uint8.astype(np.float32) * (1 - alpha_map)
        + colored.astype(np.float32) * alpha_map
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def sono_gradcam_pp(img_array):
    """
    Proper Grad-CAM++ for the Keras sonography model.
    Ported from GradCAMPP_MMBCNet_Sonography_(1).ipynb (cell 10).

    Steps
    -----
    1. Run TTA prediction for the class probabilities.
    2. Watch conv_output inside the GradientTape so second-order gradients
       are available for the alpha computation.
    3. Compute Grad-CAM++ weights:  alpha = grads² / (2·grads² + A·grads³)
    4. Produce heatmap = ReLU(Σ_k  weight_k · A_k), normalise to [0, 1].
    5. Resize to SONO_IMG_SIZE × SONO_IMG_SIZE with bilinear interpolation.
    6. Overlay with percentile-threshold / Gaussian-blur / gamma / alpha-blend.
    """
    eps = 1e-8
    img_batch = tf.cast(np.expand_dims(img_array, 0), tf.float32)

    # ── 1. TTA prediction for class probs ─────────────────────────────────────
    tta_preds = tta_predict(sono_model, img_batch)
    if tta_preds.shape[-1] == 1:
        probs = np.array([1 - float(tta_preds[0]), float(tta_preds[0])])
    else:
        probs = tta_preds[0]

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    # ── 2. Grad-CAM++ forward + gradient pass ─────────────────────────────────
    x = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # Watching conv_output (not the raw input) gives us exact Grad-CAM++
        # second-order alpha weights without needing higher-order tape nesting.
        conv_output, predictions = sono_gradcam_model(x, training=False)
        tape.watch(conv_output)
        class_score = predictions[:, pred_class]

    grads = tape.gradient(class_score, conv_output)  # (1, h, w, k)

    if grads is None:
        # Tape returned None → fall back to mean activation map
        heatmap_raw = np.mean(conv_output[0].numpy(), axis=-1)
        heatmap_raw = np.maximum(heatmap_raw, 0)
    else:
        # ── 3. Compute Grad-CAM++ alpha weights ───────────────────────────────
        conv_out = conv_output[0]   # (h, w, k)
        g        = grads[0]         # (h, w, k)

        g_sq    = tf.square(g)
        g_cube  = g_sq * g

        # Σ_spatial A_k  for each channel k
        A_sum   = tf.reduce_sum(conv_out, axis=(0, 1), keepdims=True)  # (1, 1, k)

        alpha_denom = 2.0 * g_sq + A_sum * g_cube
        alpha_denom = tf.where(
            tf.abs(alpha_denom) > eps,
            alpha_denom,
            tf.ones_like(alpha_denom) * eps,
        )
        alphas = g_sq / alpha_denom  # (h, w, k)

        # Normalise so alphas sum to 1 over spatial positions per channel
        alphas_sum = tf.reduce_sum(alphas, axis=(0, 1), keepdims=True)
        alphas_sum = tf.where(
            tf.abs(alphas_sum) > eps,
            alphas_sum,
            tf.ones_like(alphas_sum) * eps,
        )
        alphas = alphas / alphas_sum

        # Channel weights:  w_k = Σ_spatial  alpha_k · ReLU(grad_k)
        weights = tf.reduce_sum(
            alphas * tf.nn.relu(g), axis=(0, 1)
        )  # (k,)

        # ── 4. Heatmap = ReLU(Σ_k w_k · A_k) ────────────────────────────────
        heatmap_raw = tf.reduce_sum(weights * conv_out, axis=-1)  # (h, w)
        heatmap_raw = tf.nn.relu(heatmap_raw).numpy()

    # ── 5. Normalise and resize to model input resolution ─────────────────────
    max_val = float(np.max(heatmap_raw))
    if max_val > eps:
        heatmap_raw = heatmap_raw / max_val
    else:
        heatmap_raw = np.zeros_like(heatmap_raw)

    heatmap = cv2.resize(
        heatmap_raw.astype(np.float32),
        (SONO_IMG_SIZE, SONO_IMG_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    heatmap = np.clip(heatmap, 0.0, 1.0)

    # ── 6. Produce overlay with notebook-quality postprocessing ───────────────
    original_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
    cam_image = _sono_overlay_heatmap(original_uint8, heatmap)

    return cam_image, pred_class, confidence, heatmap, probs





# ══════════════════════════════════════════════════════════════════════════════
# FLAG LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def mammo_flag_logic(probs, pred_class):
    flag, reasons = False, []
    mal_spectrum = float(probs[MAMMO_IDX_SUSPMAL] + probs[MAMMO_IDX_MAL])
    if pred_class == MAMMO_IDX_SUSPMAL:
        flag = True
        reasons.append("Suspicious Malignant predicted — mandatory review")
    elif pred_class == MAMMO_IDX_BENIGN and mal_spectrum > MAMMO_SPECTRUM_THRESH:
        flag = True
        reasons.append(
            f"Benign challenged by malignant spectrum: "
            f"p(SuspMal)={probs[MAMMO_IDX_SUSPMAL]:.2f} + "
            f"p(Mal)={probs[MAMMO_IDX_MAL]:.2f} = {mal_spectrum:.2f} > {MAMMO_SPECTRUM_THRESH}"
        )
    elif pred_class == MAMMO_IDX_MAL and float(probs[pred_class]) < MAMMO_MAL_CONF_THRESH:
        flag = True
        reasons.append(
            f"Low confidence on Malignant: {probs[pred_class]:.3f} < {MAMMO_MAL_CONF_THRESH}"
        )
    return flag, reasons


def sono_flag_logic(probs, pred_class):
    flag, reasons = False, []
    malignant_idx = SONO_CLASSES.index("Malignant") if "Malignant" in SONO_CLASSES else -1
    thresh = SONO_THRESH_MALIGNANT if pred_class == malignant_idx else SONO_THRESH_DEFAULT
    if float(probs[pred_class]) < thresh:
        flag = True
        reasons.append(f"Low confidence: {probs[pred_class]:.3f} < {thresh:.2f}")
    return flag, reasons


# ══════════════════════════════════════════════════════════════════════════════
# GRADCAM IMAGE → BASE64
# ══════════════════════════════════════════════════════════════════════════════

def cam_to_base64(cam_image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(cam_image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/predict/mammo")
async def predict_mammo(file: UploadFile = File(...)):
    """Single mammogram → prediction + confidence + flag + Grad-CAM++ image (base64 PNG).
    Accepts JPEG/PNG images or DICOM (.dcm) files."""
    raw = await file.read()
    dicom_metadata = None
    content_type = (file.content_type or "").lower()
    is_std_image = content_type.startswith("image/") and "dicom" not in content_type

    # Try DICOM if: extension/magic matches, OR content-type isn't a standard image
    # (browsers often send .dcm as application/octet-stream)
    if _is_dicom(raw, file.filename or "") or not is_std_image:
        if PYDICOM_AVAILABLE:
            try:
                raw, dicom_metadata = _dicom_to_jpeg(raw)
            except Exception:
                if not is_std_image:
                    raise HTTPException(
                        status_code=400,
                        detail="Unsupported file. Upload a JPG, PNG, or DICOM (.dcm) file."
                    )
                # It looked like DICOM by name but failed; fall through to PIL
        elif not is_std_image:
            raise HTTPException(status_code=400, detail="File must be an image (JPG/PNG) or DICOM (.dcm).")

    try:
        image_np   = preprocess_mammogram_image(raw)
        img_tensor = mammo_transform(image=image_np)["image"].unsqueeze(0)
        cam_image, pred_cls, conf, _, probs = mammo_gradcam_pp(img_tensor)
        flag, reasons = mammo_flag_logic(probs, pred_cls)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({
        "modality"         : "Mammography",
        "filename"         : file.filename,
        "prediction"       : MAMMO_CLASSES[pred_cls],
        "confidence"       : round(conf, 4),
        "probabilities"    : {c: round(float(p), 4) for c, p in zip(MAMMO_CLASSES, probs)},
        "flagged"          : flag,
        "flag_reasons"     : reasons,
        "gradcam_image_b64": cam_to_base64(cam_image),
        "dicom_metadata"   : dicom_metadata,
    })


@app.post("/predict/sono")
async def predict_sono(file: UploadFile = File(...)):
    """Single sonogram → prediction + confidence + flag + Grad-CAM++ image (base64 PNG).
    Accepts JPEG/PNG images or DICOM (.dcm) files."""
    raw = await file.read()
    dicom_metadata = None
    content_type = (file.content_type or "").lower()
    is_std_image = content_type.startswith("image/") and "dicom" not in content_type

    if _is_dicom(raw, file.filename or "") or not is_std_image:
        if PYDICOM_AVAILABLE:
            try:
                raw, dicom_metadata = _dicom_to_jpeg(raw)
            except Exception:
                if not is_std_image:
                    raise HTTPException(
                        status_code=400,
                        detail="Unsupported file. Upload a JPG, PNG, or DICOM (.dcm) file."
                    )
        elif not is_std_image:
            raise HTTPException(status_code=400, detail="File must be an image (JPG/PNG) or DICOM (.dcm).")


    try:
        img_array = preprocess_sono_image(raw)
        cam_image, pred_cls, conf, _, probs = sono_gradcam_pp(img_array)
        flag, reasons = sono_flag_logic(probs, pred_cls)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({
        "modality"         : "Sonography",
        "filename"         : file.filename,
        "prediction"       : SONO_CLASSES[pred_cls],
        "confidence"       : round(conf, 4),
        "probabilities"    : {c: round(float(p), 4) for c, p in zip(SONO_CLASSES, probs)},
        "flagged"          : flag,
        "flag_reasons"     : reasons,
        "gradcam_image_b64": cam_to_base64(cam_image),
        "dicom_metadata"   : dicom_metadata,
    })


@app.post("/predict/mammo/bulk")
async def predict_mammo_bulk(files: List[UploadFile] = File(...)):
    """Multiple mammograms → JSON list of (prediction, confidence, flagged). No Grad-CAM."""
    results = []
    for f in files:
        raw = await f.read()
        try:
            image_np   = preprocess_mammogram_image(raw)
            img_tensor = mammo_transform(image=image_np)["image"].unsqueeze(0)
            with torch.no_grad():
                logits = mammo_model(img_tensor.to(DEVICE))
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_cls = int(np.argmax(probs))
            conf     = float(probs[pred_cls])
            flag, reasons = mammo_flag_logic(probs, pred_cls)
            results.append({
                "filename"     : f.filename,
                "prediction"   : MAMMO_CLASSES[pred_cls],
                "confidence"   : round(conf, 4),
                "probabilities": {c: round(float(p), 4) for c, p in zip(MAMMO_CLASSES, probs)},
                "flagged"      : flag,
                "flag_reasons" : reasons,
                "error"        : None,
            })
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})
    return JSONResponse({"modality": "Mammography", "total": len(results), "results": results})


@app.post("/predict/sono/bulk")
async def predict_sono_bulk(files: List[UploadFile] = File(...)):
    """Multiple sonograms → JSON list of (prediction, confidence, flagged). No Grad-CAM."""
    results = []
    for f in files:
        raw = await f.read()
        try:
            img_array   = preprocess_sono_image(raw)
            img_batch   = tf.cast(np.expand_dims(img_array, 0), tf.float32)
            tta_preds   = tta_predict(sono_model, img_batch)
            if tta_preds.shape[-1] == 1:
                probs = np.array([1 - float(tta_preds[0][0]), float(tta_preds[0][0])])
            else:
                probs = tf.nn.softmax(tta_preds[0]).numpy()
            pred_cls = int(np.argmax(probs))
            conf     = float(probs[pred_cls])
            flag, reasons = sono_flag_logic(probs, pred_cls)
            results.append({
                "filename"     : f.filename,
                "prediction"   : SONO_CLASSES[pred_cls],
                "confidence"   : round(conf, 4),
                "probabilities": {c: round(float(p), 4) for c, p in zip(SONO_CLASSES, probs)},
                "flagged"      : flag,
                "flag_reasons" : reasons,
                "error"        : None,
            })
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})
    return JSONResponse({"modality": "Sonography", "total": len(results), "results": results})


@app.get("/health")
def health():
    return {
        "status"        : "ok",
        "mammo_model"   : mammo_model is not None,
        "sono_model"    : sono_model is not None,
        "device"        : DEVICE,
        "sono_img_size" : SONO_IMG_SIZE,
    }