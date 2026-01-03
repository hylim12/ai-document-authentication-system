import os
import joblib
import pandas as pd
import numpy as np
import warnings

# IMPORT FEATURE EXTRACTION LOGIC 
from forged_document_detector import DocumentForgeryDetector, ensure_output_folder 

warnings.filterwarnings('ignore')

# Update this path to the image you want to test
NEW_IMAGE_PATH = 'more_docs/alb_id_41_fake_6_14.jpg'

MODEL_PATH = 'final_rf_model.joblib'
IMPORTANCE_PATH = 'feature_importances.joblib' 
CSV_HEADER_PATH = 'ml_training_data.csv' 

# PRIORITY SET TO 6
TOP_N_FEATURES = 6 

# 2. PREDICTION FUNCTION 

def generate_reasoning(single_feature_df, importances, baseline_means, detector_obj, top_n=TOP_N_FEATURES): 
    """
    Analyzes the input against baseline means and reports explicit Standard Violations
    detected by the DocumentForgeryDetector.
    """
    reasons = []
    
    # --- 1. REPORT EXPLICIT STANDARD VIOLATIONS (Supervisor Requirement) ---
    if hasattr(detector_obj, 'anomalies'):
        # Filter for high-severity standard errors flagged during processing
        std_violations = []
        for a in detector_obj.anomalies:
            for t in a['types']:
                if "STD_ERR" in t and t not in std_violations:
                    std_violations.append(t)
        
        if std_violations:
            reasons.append("!!! CRITICAL BASELINE MISMATCHES DETECTED !!!")
            for violation in std_violations:
                # Format string like STD_ERR_GIVEN NAME_TOO_LARGE to a readable reason
                readable = violation.replace("STD_ERR_", "").replace("_", " ")
                reasons.append(f"- {readable}: This field violates the official document standard.")
            reasons.append("-" * 40)

    # --- 2. ML FEATURE DEVIATION ANALYSIS ---
    input_features = single_feature_df.iloc[0].drop(['Document_ID', 'Label'], errors='ignore')
    sorted_importances = importances.sort_values(ascending=False)
    top_features = sorted_importances.head(top_n)
    
    reasons.append(f"Top {top_n} most impactful ML features analysis:")
    
    for feature_name, importance_score in top_features.items():
        input_value = input_features.get(feature_name)
        baseline_value = baseline_means.get(feature_name, 0)
        
        if input_value is None: continue
        deviation = input_value - baseline_value
        
        if 'Ratio' in feature_name or 'Count' in feature_name:
            if deviation > baseline_value * 0.25 and input_value > 0.05: 
                reasons.append(f"- HIGH ANOMALY: {feature_name} ({input_value:.4f}) is significantly above baseline ({baseline_value:.4f}).")
            else:
                reasons.append(f"- CONSISTENT: {feature_name} is within acceptable range.")
        elif 'STD' in feature_name or 'Grad' in feature_name: 
            if deviation > baseline_value * 0.5 and input_value > 1.0: 
                reasons.append(f"- HIGH VARIABILITY: {feature_name} indicates non-uniformity in ink or alignment.")
            else:
                reasons.append(f"- CONSISTENT: {feature_name} is stable.")

    return reasons

def make_single_prediction(image_path, model, importances, baseline_means, header_df):
    """Integrated prediction logic with mandatory forensic override for FYP."""
    detector = DocumentForgeryDetector(image_path)
    detector.process_document()
    
    # 1. Feature Alignment using header_df template
    features = detector.forgery_features
    new_data_df = pd.DataFrame([features])
    feature_cols = header_df.drop(['Document_ID', 'Label'], axis=1).columns.tolist()
    X_predict = new_data_df.reindex(columns=feature_cols).fillna(0).replace([np.inf, -np.inf], 0)

    # 2. Check for Forensic Standard Violations (Size/Thickness)
    has_std_error = any("STD_ERR" in str(a) for a in detector.anomalies)

    # 3. Model Prediction
    prediction = model.predict(X_predict)[0]
    probabilities = model.predict_proba(X_predict)[0]
    
    # 4. OVERRIDE: Physical standard failure dictates a FORGED verdict
    if has_std_error:
        prediction = 1 
        probabilities[1] = max(probabilities[1], 0.98) 
        detector.final_verdict = "FORGED"
    else:
        detector.final_verdict = "FORGED" if prediction == 1 else "AUTHENTIC"

    # 5. Save report with string path and corrected filenames
    ensure_output_folder()
    out_name = f"{os.path.basename(image_path).split('.')[0]}_PREDICTION_REPORT.PNG"
    output_png = os.path.join("PNG_results", out_name)
    detector.visualize_results(save_path=output_png)
    
    # Corrected variable from 'reasonings' to 'reasoning'
    reasoning = generate_reasoning(new_data_df, importances, baseline_means, detector)
    return prediction, probabilities, reasoning
    
# 3. MAIN EXECUTION 
if __name__ == "__main__":
    
    print("\n========================================================")
    print("      FINAL FORGERY PREDICTION SYSTEM STARTING")
    print("========================================================")

    try:
        rf_model = joblib.load(MODEL_PATH)
        feature_importances = joblib.load(IMPORTANCE_PATH) 
        csv_header_df = pd.read_csv(CSV_HEADER_PATH)
        
        feature_cols = feature_importances.index.tolist()
        BASELINE_DF = csv_header_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        BASELINE_MEANS = BASELINE_DF.mean().to_dict()
        
        print(f"[INFO] Model and baseline loaded successfully.")

    except Exception as e:
        print(f"[FATAL] Initialization error: {e}")
        exit()
        
    final_prediction, probabilities, prediction_reasoning = make_single_prediction(
        NEW_IMAGE_PATH, rf_model, feature_importances, BASELINE_MEANS, csv_header_df
    )
    
    if final_prediction is not None:
        verdict = "FORGED" if final_prediction == 1 else "AUTHENTIC"
        confidence = probabilities[final_prediction] * 100
        
        print("\n========================================================")
        print(f"       VERDICT for {os.path.basename(NEW_IMAGE_PATH)}")
        print("========================================================")
        print(f"FINAL PREDICTION: {verdict}")
        print(f"CONFIDENCE:       {confidence:.2f}%")
        
        print(f"\n--- FORENSIC REASONING ---")
        for reason in prediction_reasoning:
            print(reason)
            
        print("========================================================")
        print("System Demonstration Complete.")