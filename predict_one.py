import os
import joblib
import pandas as pd
import numpy as np
import warnings

from forged_document_detector import DocumentForgeryDetector, ensure_output_folder

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

# Image to be predicted (from more_docs, NOT input_docs)
NEW_IMAGE_PATH = "more_docs/alb_id_94_fake_6_50.jpg"

# Paths produced by ml_model_training.ipynb
MODEL_PATH = "trained_models/forged_document_rf_model.pkl"
FEATURE_COLUMNS_PATH = "trained_models/feature_columns.pkl"

# ============================================================
# LOAD TRAINED MODEL & FEATURE SCHEMA
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    print("[INFO] Trained model and feature schema loaded successfully.")
except Exception as e:
    raise RuntimeError(f"[FATAL] Failed to load model artifacts: {e}")

# ============================================================
# SINGLE DOCUMENT PREDICTION FUNCTION
# ============================================================

def predict_single_document(image_path):
    """
    Runs full forensic feature extraction + ML prediction
    on a single document image.
    """

    # 1. Run forensic feature extraction
    detector = DocumentForgeryDetector(image_path)
    detector.process_document()

    # 2. Convert features to DataFrame
    feature_dict = detector.forgery_features
    X_input = pd.DataFrame([feature_dict])

    # 3. Enforce exact training feature order
    X_input = X_input.reindex(columns=feature_columns, fill_value=0)
    X_input = X_input.replace([np.inf, -np.inf], 0)

    # 4. ML prediction
    pred_label = model.predict(X_input)[0]
    pred_proba = model.predict_proba(X_input)[0]

    verdict = "FORGED" if pred_label == 1 else "AUTHENTIC"
    confidence = pred_proba[pred_label] * 100

    detector.ml_verdict = verdict
    detector.ml_confidence = confidence

    # 5. Save visualization
    ensure_output_folder()
    out_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_ML_PREDICTION.png"
    output_png = os.path.join("PNG_results", out_name)
    detector.visualize_results(save_path=output_png)

    return verdict, confidence, pred_proba, detector

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("\n========================================================")
    print("      FINAL FORGED DOCUMENT ML PREDICTION SYSTEM")
    print("========================================================")

    if not os.path.exists(NEW_IMAGE_PATH):
        raise FileNotFoundError(f"Input image not found: {NEW_IMAGE_PATH}")

    verdict, confidence, probabilities, detector = predict_single_document(
        NEW_IMAGE_PATH
    )

    print(f"\nDocument: {os.path.basename(NEW_IMAGE_PATH)}")
    print(f"Final Verdict : {verdict}")
    print(f"Confidence    : {confidence:.2f}%")

    print("\nClass Probabilities:")
    print(f"  Genuine (0): {probabilities[0]:.4f}")
    print(f"  Forged  (1): {probabilities[1]:.4f}")

    print("\n[INFO] Visualization saved to PNG_results/")
    print("========================================================")
