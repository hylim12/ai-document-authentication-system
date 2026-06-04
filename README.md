# AI-Powered Document Authentication for Anti-Money Laundering (AML) Systems

**Author:** Eldeena Lim Huey Yinn  
**Supervisor:** Prof. Ts. Dr. Tee Connie  
**Institution:** Faculty of Information Science and Technology, Multimedia University (MMU)  
**Academic Year:** 2025/2026

---

## 1. Overview

This repository implements an AI-assisted identity-document authentication pipeline for AML/KYC document screening. The current codebase combines:

- **Computer vision (CV)** feature extraction for typography, ink, edge-gradient, background, OCR-box, and clustered-region anomalies.
- **PaddleOCR-based OCR** with multiple image variants to improve text and bounding-box recovery.
- **Rule-based NER / field extraction** using multilingual regular-expression labels, MRZ parsing, spatial matching, and country-specific calibration.
- **Country-aware validation** for identity fields such as nationality, document numbers, personal numbers, sex, and date ordering.
- **Leakage-aware supervised ML** using a scikit-learn Random Forest pipeline for binary classification of documents as authentic or forged.
- **Explainability outputs** including JSON OCR/field exports, feature vectors, risk scores, terminal reports, and annotated PNG visualizations.

The implementation is currently focused on documents from **Albania (ALB)**, **Latvia (LVA)**, and **Slovakia (SVK)**.

---

## 2. Current Pipeline

```text
images in datasets/{training_set,validation_set,testing_set}/
        │
        ▼
dataset_builder.py
        │  extracts CV/OCR/NER/risk features and labels from filenames
        ▼
dataset_outputs/{train.csv,val.csv,test.csv}
        │
        ▼
ml_model_training.py
        │  trains leakage-aware sklearn Pipeline(RandomForestClassifier)
        ▼
trained_models/forged_document_rf_model.pkl
trained_models/feature_preprocessor.pkl
trained_models/training_metrics.json
        │
        ▼
predict_one.py
        │  runs single-document inference with feature alignment
        ▼
ML_Verdict_PNGs/<image>_ML_PREDICTION.png
```

---

## 3. Repository Structure

| Path | Purpose |
| --- | --- |
| `feature_engineering.py` | Main document-processing engine. Loads/resizes images, performs PaddleOCR, extracts fields, validates country-specific values, detects anomalies, builds ML features, saves JSON outputs, and renders annotated PNGs. |
| `dataset_builder.py` | Batch feature-extraction script for `datasets/training_set`, `datasets/validation_set`, and `datasets/testing_set`. Produces `dataset_outputs/train.csv`, `val.csv`, and `test.csv`. |
| `ml_model_training.py` | Leakage-aware Random Forest training/evaluation script. Saves the full sklearn pipeline, preprocessor, and metrics JSON under `trained_models/`. |
| `predict_one.py` | Single-image inference script. Loads the saved Random Forest pipeline, extracts aligned features for one image, predicts `AUTHENTIC` or `FORGED`, and writes an ML verdict visualization. |
| `prompts/regex_passport_patterns.py` | Multilingual regex label patterns for field extraction. |
| `utils/calibration.py` | Entity cleanup, country-specific calibration, derived nationality handling, and rule-based risk scoring helpers. |
| `utils/validators.py` | Reusable validators for normalized nationality, personal number, document number, and sex fields. |

---

## 4. Supported Inputs and Labels

### 4.1 Dataset Folders

The scripts expect this dataset layout:

```text
datasets/
  training_set/
    *.jpg / *.jpeg / *.png
  validation_set/
    *.jpg / *.jpeg / *.png
  testing_set/
    *.jpg / *.jpeg / *.png
```

### 4.2 Label Convention

`dataset_builder.py` derives the ground-truth label from each filename:

- Filename contains `fake` or `forged` → `Label = 1` (**FORGED**)
- Otherwise → `Label = 0` (**AUTHENTIC**)

Examples:

```text
alb_id_01_fake.jpg    -> 1
lva_passport_real.jpg -> 0
svk_id_forged.png     -> 1
```

### 4.3 Supported Country Codes

The current country-aware logic maps OCR-detected country names to:

| Country | Code |
| --- | --- |
| Albania | `ALB` |
| Latvia | `LVA` |
| Slovakia | `SVK` |
| Unknown / unsupported | `UNK` |

---

## 5. Feature Engineering Summary

`DocumentForgeryDetector` in `feature_engineering.py` generates a structured feature vector that includes:

### 5.1 OCR and Field Features

- OCR mean confidence (`OCR_Confidence_Mean`)
- Detected core-field count (`Field_Detected_Count`)
- Core-field completeness ratio (`Field_Completeness_Ratio`)
- Country-specific field presence such as `Has_POB`
- Rule-based extracted fields such as surname, given name, nationality, document number, personal number, dates, sex, authority, signature, height, and MRZ lines

### 5.2 Character and Layout Features

- Character count (`Char_Count`)
- Height / width / aspect-ratio mean and standard deviation
- Font-size variance
- Ink intensity and ink-density statistics
- Edge-gradient statistics

### 5.3 Anomaly Features

- Geometric anomaly ratio
- Ink anomaly ratio
- OCR-box anomaly count
- Background anomaly line count
- Clustered suspicious-region count
- Number of character, background, and OCR-box anomalies

### 5.4 AML-Oriented Risk Features

- Composite `Risk_Score` normalized to `0..100`
- Logical date-ordering issues
- Country-specific document-field validation issues
- OCR quality and field-completeness proxy features used by dataset building and inference

---

## 6. Model Training Details

`ml_model_training.py` trains a scikit-learn `Pipeline` with:

- `ColumnTransformer` preprocessing
  - Median imputation for numeric features
  - Most-frequent imputation + one-hot encoding for `Country_Code` / `Country_Name`
- `RandomForestClassifier`
  - `n_estimators=350`
  - `random_state=42`
  - `class_weight="balanced"`

The training script includes safeguards against data leakage:

- Drops direct identity/label-leaking columns such as `Image_Name`, `Document_ID`, and `Detection_Label`
- Drops pre-encoded country columns such as `Country_Code_*` and `Country_Name_*`
- Rejects columns whose names contain suspicious label tokens such as `fake`, `real`, `forged`, or `authentic`
- Drops near-perfect single-feature separators discovered from the training split only (`AUC >= 0.995`)
- Aligns test features to the training feature set before evaluation

Evaluation metrics written to `trained_models/training_metrics.json` include:

- Accuracy
- Precision
- Recall / detection rate
- F1-score
- Specificity
- False positive rate
- ROC-AUC
- Confusion-matrix counts (`tn`, `fp`, `fn`, `tp`)
- Feature columns used by the trained model
- Dropped overpowered features

---

## 7. Setup

A minimal Python environment should include the libraries imported by the codebase:

```bash
pip install opencv-python numpy matplotlib pillow pandas scikit-learn paddleocr
```

Notes:

- `paddleocr` is optional at import time, but OCR will be unavailable if it is not installed.
- The current OCR engine is initialized with CPU mode (`use_gpu=False`).
- If your PaddleOCR installation requires PaddlePaddle separately, install the CPU or GPU PaddlePaddle package recommended for your platform before running the pipeline.

---

## 8. Usage

### 8.1 Build Datasets

```bash
python dataset_builder.py
```

Expected CSV outputs:

```text
dataset_outputs/train.csv
dataset_outputs/val.csv
dataset_outputs/test.csv
```

During processing, the detector also writes OCR/field JSON and analysis PNG artifacts for processed documents.

### 8.2 Train and Evaluate the Model

```bash
python ml_model_training.py
```

Expected model outputs:

```text
trained_models/forged_document_rf_model.pkl
trained_models/feature_preprocessor.pkl
trained_models/training_metrics.json
```

### 8.3 Run Single-Document Inference

Edit `NEW_IMAGE_PATH` in `predict_one.py` to point at the image you want to classify, then run:

```bash
python predict_one.py
```

Expected inference output:

- Terminal verdict: `AUTHENTIC` or `FORGED`
- Confidence percentage
- Class probabilities for genuine (`0`) and forged (`1`)
- Annotated visualization under `ML_Verdict_PNGs/`

---

## 9. Generated Artifacts

Depending on which scripts you run, the repository may generate:

```text
dataset_outputs/
  train.csv
  val.csv
  test.csv

trained_models/
  forged_document_rf_model.pkl
  feature_preprocessor.pkl
  training_metrics.json

final_results/PNG_results/
  <image>_analysis.png

final_results/results/OCR_JSON_results/
  <image>.json

final_results/results/FIELD_JSON_results/
  <image>.json

ML_Verdict_PNGs/
  <image>_ML_PREDICTION.png
```

---

## 10. Data Integrity Notes

The current implementation is designed to reduce leakage and keep training/inference consistent:

- Filename-derived labels are used only as the target label during dataset generation.
- Label-bearing filenames and explicit detection labels are dropped before training.
- Country is handled as a categorical feature inside the saved sklearn pipeline.
- Inference uses the saved pipeline's expected feature names and fills missing runtime features with safe defaults.
- Slovakia-specific logic disables `Has_POB` because place of birth is not expected for the supported Slovakian document configuration.

---

## 11. Known Operational Notes

- The pipeline is file-path driven; dataset paths and the sample inference image path are constants in the scripts.
- OCR quality strongly affects field extraction, risk scoring, and downstream model confidence.
- Rule-based NER is intentionally deterministic and auditable; no LLM-based NER is active in the current code.
- The model must be retrained after changing feature definitions or dataset composition.
- The current codebase supports ALB/LVA/SVK logic; new countries require updates to regex patterns, calibration rules, validators, country detection, and required/optional field definitions.

---

## 12. Recommended End-to-End Workflow

```bash
# 1. Add labeled document images to datasets/training_set, validation_set, and testing_set

# 2. Generate feature CSVs
python dataset_builder.py

# 3. Train and evaluate the model
python ml_model_training.py

# 4. Update NEW_IMAGE_PATH in predict_one.py, then run inference
python predict_one.py
```

---

## 13. Project Contribution

This project contributes a hybrid document-authentication workflow for AML use cases by combining forensic CV features, deterministic OCR/NER validation, country-aware rule checks, leakage-aware ML training, and explainable output artifacts suitable for compliance review.
