import pandas as pd


df = pd.read_csv("dataset_outputs/test.csv")

print("\n========== FINAL AML-CV EVALUATION ==========")

# ========================
# ML METRICS
# ========================
accuracy = (df["Prediction"] == df["Label"]).mean()
print(f"\nAccuracy: {accuracy:.4f}")

# ========================
# NER METRICS
# ========================
print("\nNER Performance")
print(f"Field Extraction Completeness: {df['Field_Completeness'].mean():.4f}")

# ========================
# OCR METRICS
# ========================
print("\nOCR Quality")
print(f"Text Extraction Reliability: {df['OCR_Quality'].mean():.4f}")

# ========================
# AML METRICS
# ========================
print("\nAML Metrics")
print(f"Avg Risk Score        : {df['Risk_Score'].mean():.4f}")
print(f"Risk Consistency Score: {df['Risk_Consistency'].mean():.4f}")

# ========================
# QUALITY THRESHOLD
# ========================
high_quality = (df["Field_Completeness"] > 0.8).mean()
print(f"\nHigh-quality extraction rate: {high_quality*100:.2f}%")
