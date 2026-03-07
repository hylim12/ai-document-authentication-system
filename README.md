=== VER 4 ===
Project Title: AI-Powered Document Authentication for AML Systems
Developer: Eldeena Lim Huey Yinn
Supervisor: Prof. Ts. Dr. Tee Connie
Institution: Faculty of Information Science & Technology, Multimedia University.

Project Overview
This project, developed by Eldeena Lim Huey Yinn for the Multimedia University (MMU) Session 2025/2026 Final Year Project, presents an intelligent, automated solution for document integrity assessment within Anti-Money Laundering (AML) frameworks.Traditional manual verification is prone to human fatigue and often fails to detect high-quality digital manipulations. This system bridges that gap by leveraging Computer Vision (CV) and Supervised Machine Learning to detect subtle visual anomalies in identity documents (passports, ID cards) and financial records (receipts, bank statements).

Core Features
1. Multi-Level Feature Extraction: Analyzes documents at the character, structural, and background levels.
2. Semantic Parsing (NER): Uses PaddleOCR and Named Entity Recognition to validate logical consistency (e.g., matching DOB with Passport expiry logic).
3. Forensic Anomaly Detection: Calculates statistical Z-scores for character geometry (height, width, aspect ratio) and ink intensity to find "outlier" modifications.
4. Machine Learning Classification: Employs a Random Forest model to provide a definitive "Authentic" or "Forged" verdict based on a fusion of forensic indicators.
5. Explainable Visualization: Generates reports that highlight suspicious regions in red boxes, providing transparency for compliance analysts.

Technical Stack
1. Language: Python 3.x
2. Libraries: OpenCV (Image processing & Binarization)
3. PaddleOCR (Text detection and recognition)
4. Scikit-Learn (Random Forest Classifier, data scaling)
5. Pandas/NumPy (Feature matrix manipulation)
6. Matplotlib/Seaborn (Visualization and Confusion Matrix analysis)

File Structure & Guide
1. forged_document_detector.py
- The Engine. This is the primary processing script.
- Functionality: Performs image preprocessing (CLAHE, Otsu's Binarization), segments characters, extracts forensic features, and clusters anomalies.
- Output: Generates a comprehensive "Forensic Authentication Report" and saves localized visualization maps to the PNG_results folder.
- Key Class: DocumentForgeryDetector

2. ml_model_training.ipynb
- The Brain. A Jupyter Notebook used for research and model optimization.
- Functionality: Loads the training feature matrix (ml_training_data.csv), performs a 70/30 train-test split, trains the Random Forest model, and evaluates performance.
- Results: The upgraded pipeline now supports multi-country training metadata (Country_Code) and a deterministic model-selection loop that can reach >=80% hold-out accuracy on the generated dataset split.

3. predict_one.py
- The Deployment Script. A streamlined interface for single-document inference.
- Functionality: Loads the trained RandomForest pipeline (`trained_models/forged_document_rf_model.pkl`) and runs the full CV pipeline on a single "unseen" image to provide an instant verdict with a confidence percentage.

How to Run
- Environment Setup: Ensure Python 3.x is installed along with dependencies:
pip install opencv-python paddleocr paddlepaddle scikit-learn pandas matplotlib

- Generate Dataset (multi-country): Run forged_document_detector.py to process two explicit folders:
  - Training set source: `training_set` -> `ml_training_data.csv`
  - Validation set source: `validation_set` -> `ml_validation_data.csv`
  - Test set source: `testing_set` -> `ml_test_data.csv`
  - Filenames should follow <country>_...jpg format (e.g., alb_id_00.jpg, lva_passport_01.jpg, svk_id_02.jpg).
  - The generated CSV now includes Country_Code so the model can generalize across nationalities.

- Unseen Evaluation Data: Keep `testing_set` separate from model training and validation and use it for final `ml_test_data.csv` evaluation and `predict_one.py` (single-image inference).
- Train Model: Run `python ml_model_training.py` to train a country-aware RandomForest model and save:
  - `trained_models/forged_document_rf_model.pkl`
  - `trained_models/feature_preprocessor.pkl`
  - `trained_models/training_metrics.json`

  - Single Prediction:
 `python predict_one.py` (uses `forged_document_rf_model.pkl`)

- Output Artifacts per processed file:
  - OCR JSON: `results/OCR_JSON_results/<doc_name>.json`
  - Visualization PNG: `PNG_results/<doc_name>_analysis.png`

This prototype serves as a Proof of Concept (PoC).

