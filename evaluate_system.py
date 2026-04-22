import pandas as pd


df = pd.read_csv("dataset_outputs/test.csv")

print("\n========== SYSTEM EVALUATION ==========")

# ========================
# NER PERFORMANCE
# ========================
print("\n--- NER PERFORMANCE ---")
print(f"Recall (Completeness): {df['Field_Completeness'].mean():.4f}")
print(f"Precision            : {df['Precision'].mean():.4f}")
print(f"F1-score             : {df['F1_Score'].mean():.4f}")

# ========================
# ML PERFORMANCE
# ========================
print("\n--- ML PERFORMANCE ---")
accuracy = (df["Prediction"] == df["Label"]).mean()
print(f"Accuracy             : {accuracy:.4f}")

# ========================
# AML METRICS
# ========================
print("\n--- AML QUALITY METRICS ---")

high_quality = (df["Field_Completeness"] > 0.8).mean()
print(f"High-quality extraction rate: {high_quality*100:.2f}%")

# ========================
# CV-SPECIFIC METRICS
# ========================
print("\n--- CV QUALITY METRICS ---")
print(f"Avg OCR Confidence       : {df['OCR_Quality'].mean():.4f}")

# ========================
# RISK ANALYSIS
# ========================
print("\n--- RISK ANALYSIS ---")
print(f"Avg Risk Score           : {df['Risk_Score'].mean():.4f}")
print(f"Avg Risk Consistency     : {df['Risk_Consistency'].mean():.4f}")
