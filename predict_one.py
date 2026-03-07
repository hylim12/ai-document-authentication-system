"""
Project Title: AI-POWERED DOCUMENT AUTHENTICATION FOR ANTI-MONEY LAUNDERING (AML) SYSTEMS
Created By: Eldeena Lim Huey Yinn
Student ID: 1211111904

File: predict_one.py
Functionality: Run single-image inference using the country-aware RandomForest pipeline.

"""
import os
import pickle
import pandas as pd
from forged_document_detector import DocumentForgeryDetector, ensure_output_folder, extract_country_code

# Image to be predicted (unseen documents)
NEW_IMAGE_PATH = "more_docs/alb_id_53_fake_6_46.jpg"

# Paths produced by ml_model_training.ipynb
MODEL_PATH = "trained_models/forged_document_rf_model.pkl"

# Load model artifacts
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("[INFO] Trained model pipeline loaded successfully.")
except Exception as e:
    raise RuntimeError(f"[FATAL] Failed to load model artifact: {e}")

def predict_single_document(image_path):
    """Extract features and run final forged/authentic prediction."""
    detector = DocumentForgeryDetector(image_path)
    detector.process_document()

    feature_dict = detector.forgery_features.copy()
    feature_dict["Country_Code"] = extract_country_code(os.path.basename(image_path))

    X_input = pd.DataFrame([feature_dict])

    pred_label = int(model.predict(X_input)[0])
    pred_proba = model.predict_proba(X_input)[0]

    verdict = "FORGED" if pred_label == 1 else "AUTHENTIC"
    confidence = float(pred_proba[pred_label]) * 100.0

    detector.ml_verdict = verdict
    detector.ml_confidence = confidence

    ensure_output_folder()
    out_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_ML_PREDICTION.png"
    output_png = os.path.join("PNG_results", out_name)
    detector.visualize_results(save_path=output_png)

    return verdict, confidence, pred_proba, detector

if __name__ == "__main__":
    print("\n========================================================")
    print("      FINAL FORGED DOCUMENT ML PREDICTION SYSTEM")
    print("========================================================")

    if not os.path.exists(NEW_IMAGE_PATH):
        raise FileNotFoundError(f"Input image not found: {NEW_IMAGE_PATH}")

    verdict, confidence, probabilities, _ = predict_single_document(NEW_IMAGE_PATH)
    
    print(f"\nDocument: {os.path.basename(NEW_IMAGE_PATH)}")
    print(f"Final Verdict : {verdict}")
    print(f"Confidence    : {confidence:.2f}%")

    print("\nClass Probabilities:")
    print(f"  Genuine (0): {probabilities[0]:.4f}")
    print(f"  Forged  (1): {probabilities[1]:.4f}")

    print("\n[INFO] Visualization saved to PNG_results/")
    print("========================================================")
