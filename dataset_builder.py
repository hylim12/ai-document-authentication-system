"""
File: dataset_builder.py

Purpose:
Batch process training, validation, and test datasets
using the feature engineering pipeline.
"""

import os
import pandas as pd
from feature_engineering import DocumentForgeryDetector

# Dataset paths
DATASET_ROOT = "datasets"

TRAIN_PATH = os.path.join(DATASET_ROOT, "training_set")
VAL_PATH = os.path.join(DATASET_ROOT, "validation_set")
TEST_PATH = os.path.join(DATASET_ROOT, "testing_set")

OUTPUT_DIR = "dataset_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_label_from_filename(filename):
    """
    Extract ground truth label from filename.

    Example:
    alb_id_01_fake.jpg → 1 (FORGED)
    alb_id_02_real.jpg → 0 (AUTHENTIC)
    """
    name = filename.lower()

    if "fake" in name or "forged" in name:
        return 1
    return 0


def process_dataset(folder_path, dataset_name):
    """
    Process all images in a folder and save to CSV.
    """
    print(f"\n[INFO] Processing {dataset_name} dataset...")

    dataset = []

    for file in os.listdir(folder_path):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder_path, file)

        try:
            detector = DocumentForgeryDetector(image_path)
            detector.process_document()

            features = detector.forgery_features.copy()

            # Field features
            features["Field_Count"] = len(getattr(detector, "field_entities", {}))

            country_name = detector.detect_country(getattr(detector, "ocr_full_text", ""))
            country_to_code = {"ALBANIA": "ALB", "LATVIA": "LVA", "SLOVAKIA": "SVK"}
            raw_code = country_to_code.get(country_name, "UNK")

            if country_name != "SLOVAKIA":
                features["Has_POB"] = int("PLACE OF BIRTH" in getattr(detector, "field_entities", {}))
            else:
                features["Has_POB"] = 0

            # Anomaly features
            features["Num_Anomalies"] = len(getattr(detector, "anomalies", []))
            features["Num_Background_Anomalies"] = len(getattr(detector, "background_anomalies", []))
            features["Num_OCR_Box_Anomalies"] = len(getattr(detector, "ocr_box_anomalies", []))

            # 🚀 AML + CV METRICS

            # OCR Quality (Text Extraction Reliability)
            features["OCR_Quality"] = features.get("OCR_Confidence_Mean", 0)

            # Field Completeness (Field Extraction Completeness Rate)
            features["Field_Completeness"] = features.get("Field_Completeness_Ratio", 0)

            # Add metadata
            features["Image_Name"] = file
            features["Country_Code"] = raw_code
            features["Label"] = extract_label_from_filename(file)

            dataset.append(features)

            print(f"[✓] Processed: {file}")

        except Exception as e:
            print(f"[✗] Failed: {file} | Error: {e}")

    df = pd.DataFrame(dataset)

    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}.csv")
    df.to_csv(output_path, index=False)

    print(f"\n[INFO] Saved {dataset_name} CSV → {output_path}")
    print(f"[INFO] Total samples: {len(df)}")

    return df


def main():
    print("========================================")
    print("     DATASET FEATURE EXTRACTION PIPELINE")
    print("========================================")

    train_df = process_dataset(TRAIN_PATH, "train")
    val_df = process_dataset(VAL_PATH, "val")
    test_df = process_dataset(TEST_PATH, "test")

    print("\n[INFO] All datasets processed successfully.")


if __name__ == "__main__":
    main()
