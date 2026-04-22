# AI-Powered Document Authentication for AML Systems

This project detects forged identity/financial documents using computer-vision forensics, NER-derived consistency checks, and a supervised ML classifier.

## System Workflow

1. **`dataset_builder.py`**
   - Extracts CV + NER + AML features.
   - Outputs: `dataset_outputs/train.csv`, `dataset_outputs/val.csv`, `dataset_outputs/test.csv`.

2. **`ml_model_training.ipynb`**
   - Trains a RandomForest model.
   - Performs **all evaluation** in-notebook:
     - ML metrics (Precision, Recall, F1, ROC-AUC)
     - NER metrics (Field Completeness, Precision, F1)
     - AML metrics (Risk Score, Risk Consistency)

3. **`predict_one.py`**
   - Runs inference on an unseen document.
   - Outputs prediction + visualization.

## Important Note

To prevent data leakage, features such as `Detection_Label` must **never** be included in training data, as they directly reveal the ground-truth label.

## Run

- Build datasets:
  - `python dataset_builder.py`
- Train/evaluate:
  - Open and run `ml_model_training.ipynb`
- Single-image inference:
  - `python predict_one.py`
