"""
File: dataset_builder.py

Purpose:
Batch process training, validation, and test datasets
using the feature engineering pipeline.
"""

import os
import pandas as pd
from feature_engineering import DocumentForgeryDetector, extract_country_code

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

            # Add metadata
            features["Image_Name"] = file
            features["Country_Code"] = extract_country_code(file)
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
