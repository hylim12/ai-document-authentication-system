import os
import joblib
import pandas as pd
import numpy as np
import warnings
from forged_document_test import DocumentForgeryDetector

warnings.filterwarnings('ignore')

NEW_IMAGE_PATH = 'more_docs/alb_id_41.jpg'
MODEL_PATH = 'final_rf_model.joblib'
IMPORTANCE_PATH = 'feature_importances.joblib' 
CSV_HEADER_PATH = 'ml_training_data.csv' 
TOP_N_FEATURES = 6 

def generate_reasoning(single_df, importances, baselines, top_n=TOP_N_FEATURES): 
    reasons = []
    input_vals = single_df.iloc[0]
    top_f = importances.sort_values(ascending=False).head(top_n)
    
    for f_name, _ in top_f.items():
        val = input_vals.get(f_name, 0)
        base = baselines.get(f_name, 0)
        diff = val - base
        if diff > base * 0.3:
            reasons.append(f"- ALERT: {f_name} is high ({val:.4f} vs base {base:.4f})")
        else:
            reasons.append(f"- OK: {f_name} is normal")
    return reasons

def make_prediction(img_path, model, importances, baselines, header_df):
    det = DocumentForgeryDetector(img_path)
    det.process_document()
    X = pd.DataFrame([det.forgery_features])
    cols = header_df.drop(['Document_ID', 'Label'], axis=1).columns
    X = X[cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    reasoning = generate_reasoning(X, importances, baselines)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    return pred, prob, reasoning

if __name__ == "__main__":
    try:
        if not os.path.exists(CSV_HEADER_PATH) or os.path.getsize(CSV_HEADER_PATH) == 0:
            print("[FATAL] ml_training_data.csv is empty. Run forged_document_detector.py first.")
            exit()

        model = joblib.load(MODEL_PATH)
        imps = joblib.load(IMPORTANCE_PATH)
        df_train = pd.read_csv(CSV_HEADER_PATH)
        
        # Calculate baselines from CSV
        numeric_cols = imps.index.tolist()
        baselines = df_train[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0).mean().to_dict()

        p, prob, reasons = make_prediction(NEW_IMAGE_PATH, model, imps, baselines, df_train)
        
        print(f"\nVERDICT: {'FORGED' if p == 1 else 'AUTHENTIC'}")
        print(f"CONFIDENCE: {prob[p]*100:.2f}%")
        for r in reasons: print(r)

    except Exception as e:
        print(f"Error: {e}")