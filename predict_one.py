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
from feature_engineering import DocumentForgeryDetector, ensure_output_folder, extract_country_code

# Image to be predicted (unseen documents)
NEW_IMAGE_PATH = "datasets/testing_set/alb_id_84_fake_6_111.jpg"

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
    raw_code = extract_country_code(os.path.basename(image_path))
    feature_dict["Country_Code"] = raw_code
    feature_dict["Country_Name"] = country_map.get(raw_code, "UNKNOWN")

    # Inject calibrated NER features
    if hasattr(detector, "ner_entities"):
        feature_dict["Risk_Score"] = getattr(detector, "risk_score", 0.0)
        feature_dict["NER_Field_Count"] = len(detector.ner_entities)

        # Slovakia has no PLACE OF BIRTH expected
        if raw_code != "SVK":
            feature_dict["Has_POB"] = int("PLACE OF BIRTH" in detector.ner_entities)
        else:
            feature_dict["Has_POB"] = 0

    # AML anomaly-strength features
    feature_dict["Num_Anomalies"] = len(getattr(detector, "anomalies", []))
    feature_dict["Num_Background_Anomalies"] = len(getattr(detector, "background_anomalies", []))
    feature_dict["Num_OCR_Box_Anomalies"] = len(getattr(detector, "ocr_box_anomalies", []))

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

    # Optional explainability hook
    detector.feature_snapshot = feature_dict.copy()

    # Backward/forward compatibility: align with model's expected input columns.
    expected_cols = None
    try:
        preprocessor = model.named_steps.get("preprocessor")
        if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
            expected_cols = list(preprocessor.feature_names_in_)
    except Exception:
        expected_cols = None

    if expected_cols:
        aligned_features = {}
        for col in expected_cols:
            if col in feature_dict:
                aligned_features[col] = feature_dict[col]
            elif col == "Country_Code":
                aligned_features[col] = feature_dict["Country_Code"]
            else:
                aligned_features[col] = 0.0
        X_input = pd.DataFrame([aligned_features], columns=expected_cols)
    else:
        X_input = pd.DataFrame([feature_dict])

    pred_label = int(model.predict(X_input)[0])
    pred_proba = model.predict_proba(X_input)[0]

    verdict = "FORGED" if pred_label == 1 else "AUTHENTIC"
    confidence = float(pred_proba[pred_label]) * 100.0

    detector.ml_verdict = verdict
    detector.ml_confidence = confidence

    # 🚀 Ensure output directory exists (CRITICAL FIX)
    output_dir = "PNG_results"
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

    print("\n[INFO] Visualization saved to PNG_results/")
    print("========================================================")
