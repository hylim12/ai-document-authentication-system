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
from feature_engineering import DocumentForgeryDetector, ensure_output_folder

# Image to be predicted (unseen documents)
NEW_IMAGE_PATH = "datasets/testing_set/alb_id_84_fake_6_110.jpg"

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
    country_map = {
        "ALB": "ALBANIA",
        "LVA": "LATVIA",
        "SVK": "SLOVAKIA",
    }
    country_name = detector.detect_country(getattr(detector, "ocr_full_text", ""))
    country_to_code = {"ALBANIA": "ALB", "LATVIA": "LVA", "SLOVAKIA": "SVK"}
    raw_code = country_to_code.get(country_name, "UNK")
    feature_dict["Country_Code"] = raw_code
    feature_dict["Country_Name"] = country_name if country_name != "UNKNOWN" else country_map.get(raw_code, "UNKNOWN")

    # Inject calibrated field features
    if hasattr(detector, "field_entities"):
        feature_dict["Risk_Score"] = getattr(detector, "risk_score", 0.0)
        feature_dict["Field_Count"] = len(detector.field_entities)

        # Slovakia has no PLACE OF BIRTH expected
        if country_name != "SLOVAKIA":
            feature_dict["Has_POB"] = int("PLACE OF BIRTH" in detector.field_entities)
        else:
            feature_dict["Has_POB"] = 0

    # AML anomaly-strength features
    feature_dict["Num_Anomalies"] = len(getattr(detector, "anomalies", []))
    feature_dict["Num_Background_Anomalies"] = len(getattr(detector, "background_anomalies", []))
    feature_dict["Num_OCR_Box_Anomalies"] = len(getattr(detector, "ocr_box_anomalies", []))
    feature_dict["OCR_Quality"] = feature_dict.get("OCR_Confidence_Mean", 0)
    feature_dict["Field_Completeness"] = feature_dict.get("NER_Completeness_Ratio", 0)

    # Ensure all AML features exist
    default_features = {
        "Font_Size_Variance": 0.0,
        "OCR_Confidence_Mean": 0.0,
        "Field_Blur_Variance": 0.0,
        "Risk_Score": 0.0,
        "NER_Field_Count": 0,
        "Has_POB": 0,
    }
    for k, v in default_features.items():
        feature_dict.setdefault(k, v)

    expected_features = model.named_steps["preprocessor"].feature_names_in_
    for col in expected_features:
        if col not in feature_dict:
            feature_dict[col] = 0

    # Optional explainability hook
    detector.feature_snapshot = feature_dict.copy()

    # Pass raw features directly into the saved sklearn pipeline.
    # The pipeline handles one-hot encoding + feature alignment internally.
    X_input = pd.DataFrame([feature_dict])[list(expected_features)]

    pred_label = int(model.predict(X_input)[0])
    pred_proba = model.predict_proba(X_input)[0]

    verdict = "FORGED" if pred_label == 1 else "AUTHENTIC"
    confidence = float(pred_proba[pred_label]) * 100.0

    detector.ml_verdict = verdict
    detector.ml_confidence = confidence

    # Ensure output directory exists (CRITICAL FIX)
    output_dir = "ML_Verdict_PNGs"
    os.makedirs(output_dir, exist_ok=True)

    out_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_ML_PREDICTION.png"
    output_png = os.path.join(output_dir, out_name)
    detector.visualize_results(save_path=output_png)
    print(f"[INFO] Saved visualization → {output_png}")

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

    print("\n[INFO] Visualization saved to ML_Verdict_PNGs/")
    print("========================================================")
