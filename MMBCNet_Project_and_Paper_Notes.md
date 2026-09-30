# MMBCNet Project Summary and Implementation Notes

This document compiles the verified information available in the MMBCNet workspace and turns it into paper-ready notes. It distinguishes between facts that are directly visible in the code/artifacts and points that still need experimental tables if you want to make a formal comparison claim.

## 1. Project Overview

MMBCNet is a dual-modality breast cancer decision-support system for:

- Mammography analysis
- Sonography / ultrasound analysis
- Single-image inference and bulk inference
- Explainability through Grad-CAM / Grad-CAM++ overlays
- DICOM support with metadata extraction

The system is exposed through a FastAPI backend in [main.py](main.py) and a React/Vite frontend in [src/App.tsx](src/App.tsx) and [src/Dashboard.tsx](src/Dashboard.tsx).

## 2. What Models Are Used

### Mammography model

Confirmed from [main.py](main.py) and the checkpoint file name [phase2-epoch=09-val_auc=0.9036.ckpt](phase2-epoch=09-val_auc=0.9036.ckpt):

- Framework: PyTorch Lightning + timm
- Backbone: `convnext_small`
- Classifier head: custom MLP head with layers `512 -> 256 -> 4`
- Number of classes: 4
- Class labels: `Normal`, `Benign`, `Suspicious Malignant`, `Malignant`
- Loss: cross-entropy with label smoothing `0.1`
- Optimizer: AdamW with learning rate `1e-5` and weight decay `1e-4`
- Saved validation AUC in checkpoint filename: `0.9036`

### Sonography model

Confirmed from [main.py](main.py) and [Frozen92.16.keras](Frozen92.16.keras):

- Framework: TensorFlow / Keras
- Saved model name: `DenseNet121_EfficientNetB3_Fusion`
- Input size: `300 x 300 x 3`
- Number of classes: 3
- Class labels: `Benign`, `Malignant`, `Normal`
- Output layer: `classifier` with 3 units
- Fusion layer: `feature_fusion` concatenate layer
- Compiled metrics: `accuracy` and `AUC`
- AUC configuration: `num_labels = 3`, `multi_label = True`

## 3. Pipeline Architecture

### End-to-end flow

1. User uploads a JPG, PNG, or DICOM file from the frontend.
2. The frontend sends the file to the backend endpoint matching the selected modality.
3. The backend detects whether the input is DICOM or a standard image.
4. DICOM files are decoded into image bytes and metadata fields are extracted.
5. The image is preprocessed differently for mammography and sonography.
6. The corresponding model produces class probabilities and the final class label.
7. A confidence score is computed from the predicted probability.
8. Flag logic marks uncertain or clinically concerning predictions for review.
9. Grad-CAM / Grad-CAM++ heatmaps are generated for interpretation.
10. The frontend displays the prediction, probability table, review flag, metadata, and overlay heatmap.

### Mammography preprocessing

The mammography pipeline in [main.py](main.py) performs:

- Convert image to grayscale
- Remove probable corner watermark region
- Breast crop using Otsu thresholding
- Normalize intensities to a target mean and standard deviation
- Aspect-ratio preserving resize to `512 x 512`
- Zero-padding to square canvas
- Replicate to 3 channels
- Apply ImageNet normalization before inference

### Sonography preprocessing

The sonography pipeline in [main.py](main.py) performs:

- Convert image to RGB
- Resize to the model’s input size, currently `300 x 300`
- Apply CLAHE if used in the training recipe
- Normalize to `[0, 1]`

### DICOM handling

The backend supports DICOM input by:

- Detecting `.dcm` files or DICOM magic bytes
- Reading pixel data with `pydicom`
- Applying VOI LUT when available
- Converting grayscale arrays to 3-channel JPEG-compatible images
- Extracting metadata such as patient name, patient ID, study date, modality, institution, and referring physician

## 4. Detection and Classification Logic

### Mammography

The mammography model is a 4-class classifier. The backend also adds a post-prediction safety layer:

- `Suspicious Malignant` predictions are automatically flagged for review
- `Benign` predictions are flagged if the combined malignant spectrum exceeds a threshold
- `Malignant` predictions are flagged if confidence is below `0.50`

This means the system is not only classifying, but also triaging uncertain or clinically sensitive cases.

### Sonography

The sonography model is a 3-class classifier. Its post-prediction flag logic is simpler:

- Predictions are flagged when confidence falls below a threshold
- The threshold is `0.70` by default
- For `Malignant`, the threshold is stricter at `0.60`

## 5. Explainability Layer

The project includes visual explanation methods for both modalities:

- Mammography uses `GradCAMPlusPlus`
- Sonography uses a custom Grad-CAM path based on the internal fusion layer

The frontend renders the original image alongside the heatmap, which is useful for:

- Clinical review
- Paper figures
- Qualitative validation of attention regions
- Demonstrating that the model focuses on lesion-related regions rather than background noise

## 6. What Is Verified About Performance

Only a small amount of performance information is directly present in the repository.

### Verified metric evidence

- Mammography checkpoint filename includes validation AUC `0.9036`
- The mammography Lightning module registers these evaluation metrics: `Accuracy`, `AUROC`, `F1Score`, `Recall`, `Specificity`
- The sonography Keras model was compiled with `accuracy` and `AUC`

### What is not found in the repo

I did not find a README, results table, notebook export, or paper draft containing:

- Final test accuracy
- Final test AUC for the sonography model
- Precision
- Sensitivity / recall values as concrete numbers
- Specificity values as concrete numbers
- F1-score values as concrete numbers
- Confusion matrices
- Baseline comparison tables

If you want a formal paper claim, those numbers should be added from your evaluation notebook or experiment logs.

## 7. Comparison to Existing Approaches

The repository does not contain a benchmark table, so the comparison below should be used as paper wording only after you insert your actual results.

### Claimed advantages of this system

- Dual-modality design: it supports both mammography and ultrasound rather than only one imaging type
- DICOM-aware preprocessing: the system reads clinical image files directly and extracts metadata
- Explainability: Grad-CAM / Grad-CAM++ outputs make predictions easier to audit
- Review triage: confidence-based flagging helps prioritize uncertain or high-risk cases
- Modality-specific preprocessing: each image type is handled with its own preprocessing logic instead of a one-size-fits-all pipeline

### Safe comparison language for the paper

Use wording like this only if your results table supports it:

- “MMBCNet achieved higher validation AUC than the baseline models on the mammography task.”
- “The proposed fusion-based sonography model improved discrimination performance over single-backbone alternatives.”
- “The triage layer reduced silent low-confidence predictions by routing uncertain cases for review.”

Do not state numeric superiority unless you have the baseline values in hand.

## 8. Implementation Details to Include in the Paper

This is the part that is usually useful in a methodology or implementation section.

### Model implementation

- Mammography backbone: `ConvNeXt-Small`
- Mammography head: fully connected classifier with batch normalization, GELU, and dropout
- Sonography architecture: fused feature extractor with a `DenseNet121_EfficientNetB3` style fusion design
- Input sizes: `512 x 512` for mammography, `300 x 300` for sonography
- Mammography classes: 4
- Sonography classes: 3

### Training and optimization details

- Mammography loss: cross-entropy with label smoothing `0.1`
- Mammography optimizer: AdamW
- Mammography learning rate: `1e-5`
- Mammography weight decay: `1e-4`
- Sonography compile-time metrics: accuracy and AUC

### Inference details

- Mammography uses softmax probabilities from PyTorch
- Sonography uses TTA with left-right and up-down flips before prediction
- Both pipelines return class probabilities and a final predicted label
- Confidence is computed as the probability of the top predicted class

### Explainability details

- Mammography: Grad-CAM++ on the last block of the final ConvNeXt stage
- Sonography: Grad-CAM on the internal `feature_fusion` layer when available

### System details

- Backend: FastAPI
- Frontend: React + Vite
- Image augmentation / preprocessing: Albumentations, OpenCV, PIL, scikit-image
- GPU support: CUDA if available

## 9. Suggested Results Tables for the Paper

If you want the paper to be complete, include tables like these:

### Table 1. Mammography results

- Accuracy
- AUROC
- Macro F1-score
- Recall / sensitivity
- Specificity
- Precision
- Confusion matrix

### Table 2. Sonography results

- Accuracy
- AUC
- Macro F1-score
- Recall / sensitivity
- Specificity
- Precision
- Confusion matrix

### Table 3. Baseline comparison

Compare MMBCNet against:

- A standard CNN baseline
- A single-backbone model
- A no-Grad-CAM ablation
- A no-TTA ablation for sonography

Suggested columns:

- Model
- Input modality
- Accuracy
- AUC / AUROC
- F1-score
- Recall
- Specificity
- Notes

## 10. Paper-Worthy Contribution Statement

You can adapt this sentence in the introduction or contributions section:

“We propose MMBCNet, a dual-modality breast imaging decision-support system that combines modality-specific preprocessing, deep classification, uncertainty-based triage, and Grad-CAM-style explainability to support mammography and sonography analysis in a unified workflow.”

## 11. Important Caveat

The repository currently verifies model structure and one explicit mammography validation AUC value, but it does not contain a full experiment log. If you need a final submission-quality paper, you should add the exact test metrics and baseline comparison results from your training notebook or evaluation script.
