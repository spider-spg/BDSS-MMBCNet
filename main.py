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
SONO_IMG_SIZE       = 300


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

    sono_last_conv_layer = None
    
    # Try to find the primary fusion layer first for stable gradients
    for layer in sono_model.layers:
        if layer.name == 'feature_fusion':
            sono_last_conv_layer = layer.name
            break
            
    # Fallback to the last spatial layer
    if not sono_last_conv_layer:
        for layer in reversed(sono_model.layers):
            try:
                out_shape = layer.output.shape
                if isinstance(out_shape, list):
                    out_shape = out_shape[0].shape
                
                if len(out_shape) == 4 and out_shape[1] is not None and out_shape[1] > 1:
                    sono_last_conv_layer = layer.name
                    break
            except Exception:
                continue
                
    assert sono_last_conv_layer, "No spatial feature layer found in sono model."
    print(f"[startup] Sono model loaded  | input={SONO_IMG_SIZE}px | gradcam={sono_last_conv_layer}")

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


def sono_gradcam_pp(img_array):
    img_batch = tf.cast(np.expand_dims(img_array, 0), tf.float32)
    
    tta_preds = tta_predict(sono_model, img_batch)
    if tta_preds.shape[-1] == 1:
        probs = np.array([1 - float(tta_preds[0]), float(tta_preds[0])])
    else:
        probs = tta_preds[0]
        
    pred_class  = int(np.argmax(probs))
    confidence  = float(probs[pred_class])

    grad_model = tf.keras.models.Model(
        inputs  = sono_model.inputs,
        outputs = [sono_model.get_layer('multiply_1').output, sono_model.output],
    )
    
    # Watch the input tensor; the tape records ALL ops from img_batch → conv_outputs → preds
    # This lets us call tape.gradient(score, conv_outputs) even though conv_outputs wasn't
    # explicitly watched — the tape traces back through the recorded graph.
    with tf.GradientTape() as tape:
        tape.watch(img_batch)
        conv_outputs, predictions = grad_model(img_batch)
        score = predictions[:, pred_class]
        
    grads = tape.gradient(score, conv_outputs)
    
    if grads is None or tf.reduce_max(tf.abs(grads)).numpy() == 0.0:
        # Fallback: raw activation map means
        cam_raw = np.mean(conv_outputs[0].numpy(), axis=-1)
    else:
        # Standard Grad-CAM: pool gradients per channel, weight activations
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        activations  = conv_outputs[0].numpy().copy()
        
        for i in range(pooled_grads.shape[0]):
            activations[:, :, i] *= pooled_grads[i]
        
        cam_raw = np.mean(activations, axis=-1)
        cam_raw = np.maximum(cam_raw, 0)
    
    cam_res = np.array(Image.fromarray(cam_raw.astype(np.float32)).resize(
        (SONO_IMG_SIZE, SONO_IMG_SIZE), Image.BICUBIC))
    if cam_res.max() > 0:
        cam_res = cam_res / cam_res.max()

    rgb_img = np.clip(img_array / 255.0, 0, 1).astype(np.float32)
    cam_image = show_cam_on_image(rgb_img, cam_res, use_rgb=True)
    return cam_image, pred_class, confidence, cam_res, probs





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